#!/usr/bin/env python3
"""Investigate q_proj rank-126 inflection via multi-head block structure.

q_proj has an energy curve inflection at rank 126 = 2*head_dim - 2 in all 3 LFM2
models (head_dim=64 is constant across scales). This script tests whether the
inflection originates from multi-head block structure.

Analysis (weight-only, no model inference needed):
1. Full-matrix SVD of q_proj at all attention layers
2. Per-head SVD (reshape by head blocks)
3. Head orthogonality measurement
4. k_proj/v_proj comparison
5. Cross-layer inflection consistency

Usage:
    poetry run python scripts/qproj_head_structure.py --models 350M
    poetry run python scripts/qproj_head_structure.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = {
    "350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
}

# LFM2 attention layer indices (same for all scales)
ATTN_LAYER_IDXS = [2, 5, 8, 10, 12, 14]
HEAD_DIM = 64
TARGET_RANK = 126  # = 2 * HEAD_DIM - 2


def _get_weight_matrix(layer, proj_name: str):
    """Extract weight matrix from a layer's projection."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    proj = getattr(attn, proj_name, None)
    if proj is None:
        return None
    weight = getattr(proj, "weight", None)
    return weight


def _svd_float32(weight, backend):
    """SVD on float32 copy of weight matrix."""
    w = backend.astype(weight, "float32")
    backend.eval(w)
    U, S, Vt = backend.svd(w, compute_uv=True)
    backend.eval(S)
    sv = []
    for i in range(int(S.shape[0])):
        val = float(backend.to_scalar(backend.take(S, backend.array([i]), axis=0)))
        sv.append(val)
    return sv


def _find_inflection_at_rank(inflections: list[dict], target_rank: int, tolerance: int = 0) -> dict | None:
    """Find inflection point at target rank (exact or within tolerance)."""
    best = None
    best_dist = tolerance + 1
    for ip in inflections:
        dist = abs(ip["rank"] - target_rank)
        if dist <= tolerance and dist < best_dist:
            best = ip
            best_dist = dist
    return best


def analyze_model(model_path: str, model_tag: str) -> dict | None:
    """Run q_proj head structure analysis for one model."""
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s", model_path)
        return None

    logger.info("Loading model: %s", model_tag)

    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.adapters.model_backbone import resolve_model_backbone
    from modelcypher.core.domain.geometry.spectral_capacity import (
        compute_full_energy_curve,
        find_energy_inflection_points,
    )

    backend = initialize_default_backend()
    model, _tokenizer = ModelLoader(backend).load_model(model_path)
    backbone = resolve_model_backbone(model)

    if backbone is None:
        logger.error("Cannot resolve backbone for %s", model_path)
        return None

    _embed, layers, _norm = backbone
    hidden_dim = None

    model_data = {
        "model": model_tag,
        "modelPath": model_path,
        "headDim": HEAD_DIM,
        "targetRank": TARGET_RANK,
        "attentionLayers": {},
    }

    for layer_idx in ATTN_LAYER_IDXS:
        if layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]
        logger.info("  Layer %d:", layer_idx)

        layer_data = {"layerIndex": layer_idx, "projections": {}}

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight = _get_weight_matrix(layer, proj_name)
            if weight is None:
                logger.info("    %s: not found", proj_name)
                continue

            shape = list(weight.shape)
            out_dim, in_dim = shape
            logger.info("    %s: shape %s", proj_name, shape)

            if hidden_dim is None and proj_name == "q_proj":
                hidden_dim = in_dim

            # --- Full-matrix SVD ---
            full_sv = _svd_float32(weight, backend)
            full_curve = compute_full_energy_curve(full_sv)
            full_inflections = find_energy_inflection_points(full_curve)

            # Energy at target rank
            energy_at_target = full_curve[TARGET_RANK - 1] if TARGET_RANK <= len(full_curve) else None

            # Find inflection at target rank
            ip_at_target = _find_inflection_at_rank(full_inflections, TARGET_RANK, tolerance=0)
            ip_near_target = _find_inflection_at_rank(full_inflections, TARGET_RANK, tolerance=3)

            # Prominence rank of the target inflection
            prominence_rank = None
            if ip_at_target:
                for idx, ip in enumerate(full_inflections):
                    if ip["rank"] == ip_at_target["rank"]:
                        prominence_rank = idx + 1
                        break

            proj_data = {
                "shape": shape,
                "spectrumLength": len(full_sv),
                "fullSVTop10": full_sv[:10],
                "fullSVAtTarget": full_sv[TARGET_RANK - 1] if TARGET_RANK <= len(full_sv) else None,
                "energyAtTargetRank": energy_at_target,
                "totalInflectionPoints": len(full_inflections),
                "inflectionAtTargetRank": ip_at_target,
                "inflectionNearTargetRank": ip_near_target,
                "prominenceRank": prominence_rank,
                "top5Inflections": full_inflections[:5],
            }

            # --- Per-head SVD (only for q_proj, k_proj, v_proj) ---
            if proj_name in ("q_proj", "k_proj", "v_proj"):
                n_heads = out_dim // HEAD_DIM
                if out_dim % HEAD_DIM == 0 and n_heads > 0:
                    per_head_svs = []
                    for h in range(n_heads):
                        # Extract head block: [HEAD_DIM, in_dim]
                        start = h * HEAD_DIM
                        end = start + HEAD_DIM
                        # Use take to extract rows
                        row_indices = backend.array(list(range(start, end)))
                        head_block = backend.take(
                            backend.astype(weight, "float32"),
                            row_indices,
                            axis=0,
                        )
                        backend.eval(head_block)

                        # SVD of head block
                        head_sv = _svd_float32(head_block, backend)
                        per_head_svs.append(head_sv)

                    # Concatenate and sort all per-head SVs (descending)
                    all_head_svs = []
                    for hsv in per_head_svs:
                        all_head_svs.extend(hsv)
                    all_head_svs.sort(reverse=True)

                    block_curve = compute_full_energy_curve(all_head_svs)
                    block_inflections = find_energy_inflection_points(block_curve)
                    block_ip_at_target = _find_inflection_at_rank(block_inflections, TARGET_RANK, tolerance=0)
                    block_ip_near = _find_inflection_at_rank(block_inflections, TARGET_RANK, tolerance=3)
                    block_energy_at_target = block_curve[TARGET_RANK - 1] if TARGET_RANK <= len(block_curve) else None

                    proj_data["perHead"] = {
                        "nHeads": n_heads,
                        "headDim": HEAD_DIM,
                        "svsPerHead": HEAD_DIM,  # min(HEAD_DIM, in_dim) = HEAD_DIM
                        "totalBlockSVs": len(all_head_svs),
                        "blockSVTop10": all_head_svs[:10],
                        "blockEnergyAtTarget": block_energy_at_target,
                        "blockInflectionAtTarget": block_ip_at_target,
                        "blockInflectionNearTarget": block_ip_near,
                        "totalBlockInflections": len(block_inflections),
                        "top5BlockInflections": block_inflections[:5],
                    }

                    # --- Head orthogonality ---
                    # Compute mean pairwise cosine similarity of head weight blocks
                    # ||M_i^T M_j||_F / (||M_i||_F * ||M_j||_F) for i != j
                    frob_norms = []
                    for h in range(n_heads):
                        start = h * HEAD_DIM
                        row_indices = backend.array(list(range(start, start + HEAD_DIM)))
                        block = backend.take(
                            backend.astype(weight, "float32"),
                            row_indices,
                            axis=0,
                        )
                        backend.eval(block)
                        # Frobenius norm
                        fn = math.sqrt(sum(s * s for s in per_head_svs[h]))
                        frob_norms.append(fn)

                    # Sample pairwise similarities (all pairs for small n_heads)
                    pair_sims = []
                    for i in range(n_heads):
                        for j in range(i + 1, n_heads):
                            si = i * HEAD_DIM
                            sj = j * HEAD_DIM
                            ri = backend.array(list(range(si, si + HEAD_DIM)))
                            rj = backend.array(list(range(sj, sj + HEAD_DIM)))
                            wi = backend.take(backend.astype(weight, "float32"), ri, axis=0)
                            wj = backend.take(backend.astype(weight, "float32"), rj, axis=0)
                            backend.eval(wi, wj)

                            # M_i^T @ M_j: [in_dim, in_dim] — too big, use trace trick
                            # ||M_i^T M_j||_F^2 = trace((M_i^T M_j)^T (M_i^T M_j))
                            # = trace(M_j^T M_i M_i^T M_j)
                            # = ||M_i M_j^T||_F^2 (via [HEAD_DIM, HEAD_DIM] matrix)
                            cross = backend.matmul(wi, backend.transpose(wj))  # [64, 64]
                            backend.eval(cross)
                            # Frobenius norm of cross
                            cross_sv = _svd_float32(cross, backend)
                            cross_frob = math.sqrt(sum(s * s for s in cross_sv))

                            denom = frob_norms[i] * frob_norms[j]
                            sim = cross_frob / denom if denom > 0 else 0.0
                            pair_sims.append(sim)

                    mean_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0
                    max_sim = max(pair_sims) if pair_sims else 0.0
                    min_sim = min(pair_sims) if pair_sims else 0.0

                    proj_data["headOrthogonality"] = {
                        "nPairs": len(pair_sims),
                        "meanPairwiseSimilarity": mean_sim,
                        "maxPairwiseSimilarity": max_sim,
                        "minPairwiseSimilarity": min_sim,
                    }

                    logger.info("      heads=%d, mean_sim=%.4f, max_sim=%.4f",
                                n_heads, mean_sim, max_sim)

            # Log key results
            ip_str = "EXACT" if ip_at_target else ("NEAR" if ip_near_target else "NONE")
            logger.info("    %s: inflection@%d=%s, energy=%.4f, total_inflections=%d",
                        proj_name, TARGET_RANK, ip_str,
                        energy_at_target or 0.0, len(full_inflections))

            layer_data["projections"][proj_name] = proj_data

        model_data["attentionLayers"][str(layer_idx)] = layer_data

    model_data["hiddenDim"] = hidden_dim

    return model_data


def print_summary(all_results: dict[str, dict]) -> None:
    """Print cross-model summary of q_proj inflection at rank 126."""
    print()
    print("=" * 100)
    print(f"Q_PROJ HEAD STRUCTURE — INFLECTION AT RANK {TARGET_RANK} (= 2 × {HEAD_DIM} - 2)")
    print("=" * 100)

    # Cross-layer inflection table
    print()
    print("INFLECTION AT RANK 126 — ALL ATTENTION LAYERS")
    print(f"{'Model':>5} {'Layer':>5} {'Proj':>6} {'Match':>7} {'Energy':>8} {'Prom.':>8} "
          f"{'BlockMatch':>10} {'MeanSim':>8}")
    print("-" * 80)

    for model_tag in ["350M", "700M", "1.2B"]:
        data = all_results.get(model_tag)
        if data is None:
            continue

        for layer_idx in ATTN_LAYER_IDXS:
            layer_data = data.get("attentionLayers", {}).get(str(layer_idx))
            if layer_data is None:
                continue

            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = layer_data.get("projections", {}).get(proj_name)
                if proj is None:
                    continue

                ip = proj.get("inflectionAtTargetRank")
                match = "EXACT" if ip else "none"
                energy = proj.get("energyAtTargetRank", 0) or 0
                prom = proj.get("prominenceRank")
                prom_str = f"{prom}/{proj['totalInflectionPoints']}" if prom else "-"

                per_head = proj.get("perHead", {})
                block_ip = per_head.get("blockInflectionAtTarget")
                block_match = "EXACT" if block_ip else "none"

                ortho = proj.get("headOrthogonality", {})
                mean_sim = ortho.get("meanPairwiseSimilarity", 0)

                print(f"{model_tag:>5} {layer_idx:>5} {proj_name:>6} {match:>7} "
                      f"{energy:>8.4f} {prom_str:>8} {block_match:>10} {mean_sim:>8.4f}")

    # Head orthogonality summary
    print()
    print("HEAD ORTHOGONALITY (mean pairwise similarity — 0 = orthogonal, 1 = identical)")
    for model_tag in ["350M", "700M", "1.2B"]:
        data = all_results.get(model_tag)
        if data is None:
            continue
        print(f"\n  {model_tag} (hidden={data.get('hiddenDim')}):")
        for layer_idx in ATTN_LAYER_IDXS:
            layer_data = data.get("attentionLayers", {}).get(str(layer_idx))
            if layer_data is None:
                continue
            parts = []
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                proj = layer_data.get("projections", {}).get(proj_name)
                if proj and "headOrthogonality" in proj:
                    o = proj["headOrthogonality"]
                    parts.append(f"{proj_name}={o['meanPairwiseSimilarity']:.4f}")
            if parts:
                print(f"    L{layer_idx}: {', '.join(parts)}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="q_proj head structure investigation")
    parser.add_argument("--output", type=str, default="data/experiments")
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()), default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tags = args.models or list(MODELS.keys())

    all_results: dict[str, dict] = {}
    for tag in model_tags:
        result = analyze_model(MODELS[tag], tag)
        if result is not None:
            all_results[tag] = result
            tag_lower = tag.lower().replace(".", "p")
            out_path = output_dir / f"qproj_head_structure_{tag_lower}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("  Saved: %s", out_path)

    if not all_results:
        logger.error("No models analyzed")
        sys.exit(1)

    print_summary(all_results)


if __name__ == "__main__":
    main()
