#!/usr/bin/env python3
"""KVSlimmer spectral asymmetry validation (arXiv:2603.00907).

Tests whether Q/K projection weights have concentrated spectral energy while V
projection weights have dispersed spectra. Weight-only analysis — no inference.

Pre-registered predictions (see plan: refactored-gliding-graham.md):
  P1: shannon_eff_rank(V) / min_dim(V) > shannon_eff_rank(Q) / min_dim(Q) per layer
  P2: |cap_util(Q) - cap_util(K)| < |cap_util(V) - cap_util(Q)| per layer
  P3: Per-head Q/K effective rank < per-head V effective rank
  P4: Asymmetry varies with layer depth (early/highway/late)

Falsifier: >25% of attention layers across 2+ architecture families violate P1.

Usage:
    poetry run python scripts/kvslimmer_spectral_asymmetry.py \
        --model /Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16
    poetry run python scripts/kvslimmer_spectral_asymmetry.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = {
    "Qwen3.5-0.8B": "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16",
    "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "LFM2-700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "Qwen3.5-2B": "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-2B-bf16",
}

# LFM2 attention layer indices (ShortConv layers have no Q/K/V)
LFM2_ATTN_LAYER_IDXS = [2, 5, 8, 10, 12, 14]

PROJECTIONS = ("q_proj", "k_proj", "v_proj")


@dataclass
class ProjectionReport:
    """Spectral profile for one projection weight matrix."""

    proj_name: str
    shape: tuple[int, int]
    min_dim: int
    shannon_effective_rank: float
    stable_rank: float
    capacity_utilization: float
    decay_type: str
    energy_top_10pct: float
    energy_top_20pct: float
    energy_top_50pct: float
    numerical_rank_f32: int
    # Normalized metrics for cross-projection comparison
    normalized_shannon: float  # shannon_effective_rank / min_dim
    normalized_stable: float  # stable_rank / min_dim

    def to_dict(self) -> dict:
        return {
            "projName": self.proj_name,
            "shape": list(self.shape),
            "minDim": self.min_dim,
            "shannonEffectiveRank": self.shannon_effective_rank,
            "stableRank": self.stable_rank,
            "capacityUtilization": self.capacity_utilization,
            "decayType": self.decay_type,
            "energyTop10pct": self.energy_top_10pct,
            "energyTop20pct": self.energy_top_20pct,
            "energyTop50pct": self.energy_top_50pct,
            "numericalRankF32": self.numerical_rank_f32,
            "normalizedShannon": self.normalized_shannon,
            "normalizedStable": self.normalized_stable,
        }


@dataclass
class LayerAsymmetry:
    """Per-layer Q/K/V spectral asymmetry results."""

    layer_idx: int
    attention_type: str  # "full_attention", "linear_attention", "lfm2_attention"
    q: ProjectionReport
    k: ProjectionReport
    v: ProjectionReport
    # P1: V more dispersed than Q and K
    p1_v_gt_q: bool
    p1_v_gt_k: bool
    p1_pass: bool
    # P2: Q/K similar, V diverges
    qk_distance: float  # |cap_util(Q) - cap_util(K)|
    qv_distance: float  # |cap_util(V) - cap_util(Q)|
    p2_pass: bool
    # Layer depth fraction for P4
    depth_fraction: float

    def to_dict(self) -> dict:
        return {
            "layerIdx": self.layer_idx,
            "attentionType": self.attention_type,
            "q": self.q.to_dict(),
            "k": self.k.to_dict(),
            "v": self.v.to_dict(),
            "p1_v_gt_q": self.p1_v_gt_q,
            "p1_v_gt_k": self.p1_v_gt_k,
            "p1_pass": self.p1_pass,
            "qk_distance": self.qk_distance,
            "qv_distance": self.qv_distance,
            "p2_pass": self.p2_pass,
            "depthFraction": self.depth_fraction,
        }


def _get_weight_matrix(layer, proj_name: str):
    """Extract weight matrix from a layer's self_attn projection."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    proj = getattr(attn, proj_name, None)
    if proj is None:
        return None
    return getattr(proj, "weight", None)


def _detect_attention_type(layer) -> str | None:
    """Detect the attention mechanism type of a layer."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    cls_name = type(attn).__name__.lower()
    if "linear" in cls_name:
        return "linear_attention"
    # Check for LFM2 attention flag
    if hasattr(layer, "is_attention_layer"):
        return "lfm2_attention" if layer.is_attention_layer else None
    # Default: if it has q_proj, it's full attention
    if hasattr(attn, "q_proj"):
        return "full_attention"
    return None


def _analyze_projection(proj_name: str, weight, analyzer) -> ProjectionReport:
    """Run SpectralCapacityAnalyzer on a single projection weight."""
    report = analyzer.analyze(proj_name, weight)
    min_dim = min(report.weight_shape)
    shannon = report.shannon_effective_rank or 0.0
    return ProjectionReport(
        proj_name=proj_name,
        shape=report.weight_shape,
        min_dim=min_dim,
        shannon_effective_rank=shannon,
        stable_rank=report.stable_rank,
        capacity_utilization=report.capacity_utilization,
        decay_type=report.decay_type.value if report.decay_type else "unknown",
        energy_top_10pct=report.energy_fractions.top_10pct if report.energy_fractions else 0.0,
        energy_top_20pct=report.energy_fractions.top_20pct if report.energy_fractions else 0.0,
        energy_top_50pct=report.energy_fractions.top_50pct if report.energy_fractions else 0.0,
        numerical_rank_f32=report.numerical_rank_f32,
        normalized_shannon=shannon / min_dim if min_dim > 0 else 0.0,
        normalized_stable=report.stable_rank / min_dim if min_dim > 0 else 0.0,
    )


def _get_attention_layer_indices(layers, model_tag: str) -> list[int]:
    """Determine which layers have attention (architecture-specific)."""
    if "LFM2" in model_tag:
        return [i for i in LFM2_ATTN_LAYER_IDXS if i < len(layers)]

    # For Qwen and other models: check all layers for Q/K/V presence
    indices = []
    for i, layer in enumerate(layers):
        if _get_weight_matrix(layer, "q_proj") is not None:
            indices.append(i)
    return indices


def analyze_model(model_path: str, model_tag: str) -> dict | None:
    """Run Q/K/V spectral asymmetry analysis for one model."""
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s", model_path)
        return None

    logger.info("Loading model: %s (%s)", model_tag, model_path)

    from modelcypher.adapters.model_backbone import resolve_model_backbone
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain.geometry.spectral_capacity import (
        SpectralCapacityAnalyzer,
    )

    backend = initialize_default_backend()
    model, _tokenizer = ModelLoader(backend).load_model(model_path)
    backbone = resolve_model_backbone(model)

    if backbone is None:
        logger.error("Cannot resolve backbone for %s", model_path)
        return None

    _embed, layers, _norm = backbone
    total_layers = len(layers)
    analyzer = SpectralCapacityAnalyzer(backend)

    attn_indices = _get_attention_layer_indices(layers, model_tag)
    logger.info("  Found %d attention layers: %s", len(attn_indices), attn_indices)

    layer_results: list[LayerAsymmetry] = []

    for layer_idx in attn_indices:
        layer = layers[layer_idx]
        attn_type = _detect_attention_type(layer) or "unknown"
        depth_fraction = layer_idx / max(total_layers - 1, 1)

        logger.info("  Layer %d (%s, depth=%.2f):", layer_idx, attn_type, depth_fraction)

        # Analyze Q, K, V projections
        projections: dict[str, ProjectionReport] = {}
        for proj_name in PROJECTIONS:
            weight = _get_weight_matrix(layer, proj_name)
            if weight is None:
                logger.warning("    %s: not found, skipping layer", proj_name)
                break
            report = _analyze_projection(proj_name, weight, analyzer)
            projections[proj_name] = report
            logger.info(
                "    %s: shape=%s shannon=%.1f stable=%.1f cap_util=%.4f decay=%s top10%%=%.4f",
                proj_name,
                report.shape,
                report.shannon_effective_rank,
                report.stable_rank,
                report.capacity_utilization,
                report.decay_type,
                report.energy_top_10pct,
            )

        if len(projections) < 3:
            continue

        q = projections["q_proj"]
        k = projections["k_proj"]
        v = projections["v_proj"]

        # P1: V normalized shannon > Q and K normalized shannon
        p1_v_gt_q = v.normalized_shannon > q.normalized_shannon
        p1_v_gt_k = v.normalized_shannon > k.normalized_shannon
        p1_pass = p1_v_gt_q and p1_v_gt_k

        # P2: Q-K distance < Q-V distance
        qk_dist = abs(q.capacity_utilization - k.capacity_utilization)
        qv_dist = abs(v.capacity_utilization - q.capacity_utilization)
        p2_pass = qk_dist < qv_dist

        result = LayerAsymmetry(
            layer_idx=layer_idx,
            attention_type=attn_type,
            q=q,
            k=k,
            v=v,
            p1_v_gt_q=p1_v_gt_q,
            p1_v_gt_k=p1_v_gt_k,
            p1_pass=p1_pass,
            qk_distance=qk_dist,
            qv_distance=qv_dist,
            p2_pass=p2_pass,
            depth_fraction=depth_fraction,
        )
        layer_results.append(result)

        status_p1 = "PASS" if p1_pass else "FAIL"
        status_p2 = "PASS" if p2_pass else "FAIL"
        logger.info(
            "    P1=%s (V_norm=%.4f Q_norm=%.4f K_norm=%.4f) P2=%s (QK=%.4f QV=%.4f)",
            status_p1,
            v.normalized_shannon,
            q.normalized_shannon,
            k.normalized_shannon,
            status_p2,
            qk_dist,
            qv_dist,
        )

    # Aggregate predictions
    n_layers = len(layer_results)
    p1_pass_count = sum(1 for r in layer_results if r.p1_pass)
    p2_pass_count = sum(1 for r in layer_results if r.p2_pass)
    p1_fail_rate = 1.0 - (p1_pass_count / n_layers) if n_layers > 0 else 1.0
    p2_fail_rate = 1.0 - (p2_pass_count / n_layers) if n_layers > 0 else 1.0

    # P4: Depth-dependent asymmetry (compute asymmetry ratio per layer)
    # asymmetry_ratio = V_normalized_shannon / mean(Q_normalized_shannon, K_normalized_shannon)
    depth_asymmetry = []
    for r in layer_results:
        qk_mean = (r.q.normalized_shannon + r.k.normalized_shannon) / 2.0
        ratio = r.v.normalized_shannon / qk_mean if qk_mean > 0 else 0.0
        depth_asymmetry.append({
            "layerIdx": r.layer_idx,
            "depthFraction": r.depth_fraction,
            "asymmetryRatio": ratio,
            "attentionType": r.attention_type,
        })

    # Spearman rank correlation for P4 (asymmetry vs depth)
    spearman_rho = _spearman(
        [d["depthFraction"] for d in depth_asymmetry],
        [d["asymmetryRatio"] for d in depth_asymmetry],
    )

    summary = {
        "model": model_tag,
        "modelPath": model_path,
        "totalLayers": total_layers,
        "attentionLayers": n_layers,
        "predictions": {
            "P1_concentration_asymmetry": {
                "passCount": p1_pass_count,
                "totalLayers": n_layers,
                "failRate": p1_fail_rate,
                "verdict": "PASS" if p1_fail_rate <= 0.25 else "FAIL",
                "falsifierThreshold": 0.25,
            },
            "P2_qk_similarity": {
                "passCount": p2_pass_count,
                "totalLayers": n_layers,
                "failRate": p2_fail_rate,
                "verdict": "PASS" if p2_fail_rate <= 0.25 else "FAIL",
                "falsifierThreshold": 0.25,
            },
            "P4_depth_dependent": {
                "spearmanRho": spearman_rho,
                "depthAsymmetry": depth_asymmetry,
                "verdict": "PASS" if spearman_rho is not None and abs(spearman_rho) > 0.3 else "INCONCLUSIVE",
                "note": "Requires visual inspection of depth profile",
            },
        },
        "layers": [r.to_dict() for r in layer_results],
    }

    return summary


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    if n < 3:
        return None

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_rx = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    std_ry = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if std_rx == 0 or std_ry == 0:
        return 0.0
    return cov / (std_rx * std_ry)


def print_summary(results: dict) -> None:
    """Print human-readable summary table."""
    model = results["model"]
    preds = results["predictions"]

    print()
    print("=" * 110)
    print(f"KVSlimmer SPECTRAL ASYMMETRY — {model}")
    print(f"  Attention layers: {results['attentionLayers']} / {results['totalLayers']}")
    print("=" * 110)

    # Per-layer table
    print()
    print(f"{'Layer':>5} {'Type':>16} {'Depth':>5}  "
          f"{'Q_shan':>8} {'K_shan':>8} {'V_shan':>8}  "
          f"{'Q_norm':>7} {'K_norm':>7} {'V_norm':>7}  "
          f"{'P1':>4} {'P2':>4}  "
          f"{'Q_top10':>7} {'K_top10':>7} {'V_top10':>7}  "
          f"{'Q_decay':>12} {'V_decay':>12}")
    print("-" * 110)

    for layer in results["layers"]:
        q = layer["q"]
        k = layer["k"]
        v = layer["v"]
        print(
            f"{layer['layerIdx']:>5} {layer['attentionType']:>16} {layer['depthFraction']:>5.2f}  "
            f"{q['shannonEffectiveRank']:>8.1f} {k['shannonEffectiveRank']:>8.1f} {v['shannonEffectiveRank']:>8.1f}  "
            f"{q['normalizedShannon']:>7.4f} {k['normalizedShannon']:>7.4f} {v['normalizedShannon']:>7.4f}  "
            f"{'OK' if layer['p1_pass'] else 'XX':>4} {'OK' if layer['p2_pass'] else 'XX':>4}  "
            f"{q['energyTop10pct']:>7.4f} {k['energyTop10pct']:>7.4f} {v['energyTop10pct']:>7.4f}  "
            f"{q['decayType']:>12} {v['decayType']:>12}"
        )

    # Prediction verdicts
    print()
    print("PREDICTION VERDICTS")
    print("-" * 60)
    for pred_name, pred in preds.items():
        verdict = pred["verdict"]
        if "passCount" in pred:
            print(f"  {pred_name}: {verdict} ({pred['passCount']}/{pred['totalLayers']} layers pass, "
                  f"fail_rate={pred['failRate']:.2f}, threshold={pred['falsifierThreshold']})")
        elif "spearmanRho" in pred:
            rho = pred["spearmanRho"]
            rho_str = f"{rho:.3f}" if rho is not None else "N/A"
            print(f"  {pred_name}: {verdict} (Spearman rho={rho_str})")

    # Depth asymmetry profile for P4
    p4 = preds.get("P4_depth_dependent", {})
    depth_data = p4.get("depthAsymmetry", [])
    if depth_data:
        print()
        print("DEPTH ASYMMETRY PROFILE (V_norm / mean(Q_norm, K_norm))")
        print(f"{'Layer':>5} {'Depth':>5} {'Ratio':>7} {'Bar'}")
        print("-" * 50)
        for d in depth_data:
            bar_len = int(d["asymmetryRatio"] * 20)
            bar = "#" * min(bar_len, 40)
            print(f"{d['layerIdx']:>5} {d['depthFraction']:>5.2f} {d['asymmetryRatio']:>7.3f} {bar}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KVSlimmer spectral asymmetry validation (arXiv:2603.00907)"
    )
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--tag", type=str, help="Model tag (for naming output)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all registered models (smallest first)",
    )
    parser.add_argument("--output", type=str, default="data/experiments")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        model_list = list(MODELS.items())
    elif args.model:
        tag = args.tag or Path(args.model).name
        model_list = [(tag, args.model)]
    else:
        parser.error("Provide --model or --all")
        return

    all_results: dict[str, dict] = {}

    for tag, path in model_list:
        result = analyze_model(path, tag)
        if result is None:
            continue

        all_results[tag] = result
        print_summary(result)

        safe_tag = tag.lower().replace(".", "p").replace("-", "_")
        out_path = output_dir / f"kvslimmer_spectral_asymmetry_{safe_tag}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: %s", out_path)

    if not all_results:
        logger.error("No models analyzed")
        sys.exit(1)

    # Cross-model summary
    if len(all_results) >= 2:
        print()
        print("=" * 80)
        print("CROSS-MODEL FALSIFICATION CHECK")
        print("=" * 80)
        for tag, result in all_results.items():
            p1 = result["predictions"]["P1_concentration_asymmetry"]
            p2 = result["predictions"]["P2_qk_similarity"]
            print(f"  {tag}: P1={p1['verdict']} (fail_rate={p1['failRate']:.2f}) "
                  f"P2={p2['verdict']} (fail_rate={p2['failRate']:.2f})")

        # Check cross-architecture falsifier
        p1_failures = sum(
            1 for r in all_results.values()
            if r["predictions"]["P1_concentration_asymmetry"]["verdict"] == "FAIL"
        )
        if p1_failures >= 2:
            print("\n  FALSIFIER TRIGGERED: P1 fails on 2+ architectures. STOP.")
        elif p1_failures == 1:
            print("\n  WARNING: P1 fails on 1 architecture. MECHANISM_UNDERSPECIFIED.")
        else:
            print("\n  P1 SURVIVES cross-architecture check. Proceed to Phase 2.")


if __name__ == "__main__":
    main()
