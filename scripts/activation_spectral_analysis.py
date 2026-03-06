#!/usr/bin/env python3
"""Part B: Activation-space spectral analysis at layers 7,8.

Collects hidden-state activations per domain group using atlas probes,
computes SVD via Gram eigendecomposition, then analyzes energy curves
and inflection points. Compares with weight-space findings from Part A.

Usage:
    poetry run python scripts/activation_spectral_analysis.py --models 350M
    poetry run python scripts/activation_spectral_analysis.py
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

DOMAIN_RANKS = {126: "linguistic/mental", 211: "computational/structural", 255: "factual"}
TARGET_LAYERS = [7, 8]

# Domain groups: name -> list of AtlasDomain values to include
DOMAIN_GROUPS = {
    "linguistic_mental": ["linguistic", "mental"],
    "computational_structural": ["computational", "structural"],
    "mathematical_logical": ["mathematical", "logical"],
    "all_nonfactual": None,  # Everything except factual
}


def _group_probes(probes: list) -> dict[str, list]:
    """Group probes by domain group. Each probe used once (first support_text)."""
    from modelcypher.core.domain.domains import AtlasDomain

    groups: dict[str, list] = {}
    for gname, domain_values in DOMAIN_GROUPS.items():
        if domain_values is None:
            # All non-factual
            gprobes = [p for p in probes if p.domain != AtlasDomain.FACTUAL and p.support_texts]
        else:
            target_domains = {AtlasDomain(v) for v in domain_values}
            gprobes = [p for p in probes if p.domain in target_domains and p.support_texts]
        groups[gname] = gprobes
        logger.info("  Group %s: %d probes", gname, len(gprobes))

    return groups


def _forward_to_layer(embed_tokens, layers, input_ids, target_layer: int, backend):
    """Forward pass with LFM2-compatible mask routing.

    LFM2 layers expect string "causal" for attention, None for conv.
    Using numeric causal masks causes shape broadcasting errors.
    """
    hidden = embed_tokens(input_ids)
    backend.eval(hidden)

    for i, layer in enumerate(layers):
        is_attn = getattr(layer, "is_attention_layer", True)
        mask = "causal" if is_attn else None
        hidden = layer(hidden, mask=mask)
        backend.eval(hidden)
        if i == target_layer:
            break

    return hidden


def _collect_activations(
    model, tokenizer, backbone, backend, probes: list, target_layer: int,
) -> list[list[float]]:
    """Collect mean-pooled hidden states at target_layer for each probe.

    Returns backend array [n_probes, hidden_dim].
    """
    embed_tokens, layers, norm = backbone
    vecs = []

    for i, probe in enumerate(probes):
        text = probe.support_texts[0]
        tokens = tokenizer.encode(text)
        input_ids = backend.array([tokens])

        hidden = _forward_to_layer(
            embed_tokens, layers, input_ids, target_layer, backend,
        )
        # hidden: [1, seq_len, hidden_dim]

        # Mean-pool over sequence dim: [1, seq_len, hidden_dim] -> [1, hidden_dim]
        h = backend.mean(hidden, axis=1)
        backend.eval(h)
        vecs.append(h)

        if (i + 1) % 100 == 0:
            logger.info("      %d/%d probes collected", i + 1, len(probes))

    # Stack: [n_probes, hidden_dim]
    stacked = backend.concatenate(vecs, axis=0)  # [n, hidden_dim]
    backend.eval(stacked)

    return stacked


def _gram_svd(acts_arr, backend, center: bool = True) -> list[float]:
    """Compute singular values via Gram matrix eigendecomposition.

    acts_arr: Backend array [n_probes, hidden_dim].
    center: If True, subtract mean activation (removes DC component).
    Returns singular values in descending order.

    Uses A @ A^T (Gram, [n x n]) instead of full SVD on [n x d] to avoid
    MLX SVD crash on wide matrices.
    """
    n = int(acts_arr.shape[0])
    if n == 0:
        return []

    if center:
        # Remove DC component: center activations around their mean
        mean_vec = backend.mean(acts_arr, axis=0)  # [hidden_dim]
        backend.eval(mean_vec)
        # Broadcast subtract: [n, d] - [d] -> [n, d]
        acts_arr = acts_arr - mean_vec
        backend.eval(acts_arr)

    # Gram matrix: [n, n] — much smaller than [n, d]
    gram = backend.matmul(acts_arr, backend.transpose(acts_arr))
    backend.eval(gram)

    # eigvalsh returns eigenvalues in ascending order (cpu stream for MLX)
    eigvals = backend.eigvalsh(gram)
    backend.eval(eigvals)

    # Convert to list and reverse (descending)
    eig_list = [float(backend.to_scalar(backend.take(eigvals, backend.array([i]), axis=0)))
                for i in range(int(eigvals.shape[0]))]
    eig_list.reverse()

    # Singular values = sqrt(max(0, eigenvalue))
    sv = [math.sqrt(max(0.0, e)) for e in eig_list]

    return sv


def analyze_model(model_path: str, model_tag: str, probe_groups: dict[str, list]) -> dict | None:
    """Run activation spectral analysis for one model."""
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
    model, tokenizer = ModelLoader(backend).load_model(model_path)
    backbone = resolve_model_backbone(model)

    if backbone is None:
        logger.error("Cannot resolve backbone for %s", model_path)
        return None

    model_data: dict = {
        "model": model_tag,
        "modelPath": model_path,
        "groups": {},
    }

    for gname, probes in probe_groups.items():
        if not probes:
            continue

        logger.info("  Group %s (%d probes)", gname, len(probes))
        group_data: dict = {
            "probeCount": len(probes),
            "layers": {},
        }

        for target_layer in TARGET_LAYERS:
            logger.info("    Layer %d: collecting activations...", target_layer)

            acts = _collect_activations(
                model, tokenizer, backbone, backend, probes, target_layer,
            )

            logger.info("    Layer %d: computing Gram SVD (%s)...",
                        target_layer, list(acts.shape))

            sv = _gram_svd(acts, backend)
            curve = compute_full_energy_curve(sv)
            inflections = find_energy_inflection_points(curve)

            # Domain rank overlay
            domain_overlay: dict[str, dict] = {}
            for rank, domain in DOMAIN_RANKS.items():
                if rank > len(sv):
                    domain_overlay[str(rank)] = {"domain": domain, "withinSpectrum": False}
                    continue

                energy = curve[rank - 1] if rank <= len(curve) else None

                # Gap ratio at domain rank
                gap_ratio = sv[rank - 1] / sv[rank] if rank < len(sv) and sv[rank] > 0 else None

                # Nearest inflection point
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
                    "gapRatio": gap_ratio,
                    "nearestInflectionRank": nearest,
                    "distanceToInflection": nearest_dist,
                }

            # Effective rank (Shannon entropy method)
            total_energy = sum(s * s for s in sv)
            eff_rank = 0.0
            if total_energy > 0:
                probs = [(s * s) / total_energy for s in sv if s > 0]
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                eff_rank = math.exp(entropy)

            layer_data = {
                "targetLayer": target_layer,
                "matrixShape": list(acts.shape),
                "spectrumLength": len(sv),
                "effectiveRank": eff_rank,
                "singularValues": sv[:50],  # Top 50 only (save space)
                "energyCurve": curve,
                "inflectionPoints": inflections,
                "domainRankOverlay": domain_overlay,
            }

            group_data["layers"][str(target_layer)] = layer_data

            logger.info("    Layer %d: eff_rank=%.1f, %d inflections, spectrum_len=%d",
                        target_layer, eff_rank, len(inflections), len(sv))

        model_data["groups"][gname] = group_data

    return model_data


def print_summary(results: dict[str, dict]) -> None:
    """Print summary of activation spectral analysis."""
    print()
    print("=" * 90)
    print("ACTIVATION SPECTRAL ANALYSIS — DOMAIN RANK ALIGNMENT")
    print("=" * 90)

    for model_tag, model_data in results.items():
        print(f"\n{'─' * 90}")
        print(f"Model: {model_tag}")

        for gname, group_data in model_data.get("groups", {}).items():
            n_probes = group_data["probeCount"]

            for layer_idx in TARGET_LAYERS:
                layer_key = str(layer_idx)
                ld = group_data.get("layers", {}).get(layer_key)
                if ld is None:
                    continue

                print(f"\n  {gname} ({n_probes} probes) @ Layer {layer_idx}:")
                print(f"    Shape: {ld['matrixShape']}  Eff.rank={ld['effectiveRank']:.1f}  "
                      f"Inflections={len(ld['inflectionPoints'])}")

                top3 = ld["inflectionPoints"][:3]
                top3_ranks = [ip["rank"] for ip in top3]
                print(f"    Top 3 inflection ranks: {top3_ranks}")

                print(f"    Domain rank alignment:")
                for rank_key, overlay in ld["domainRankOverlay"].items():
                    if not overlay.get("withinSpectrum"):
                        print(f"      rank {rank_key} ({overlay['domain']}): outside spectrum")
                        continue
                    energy = overlay.get("energyAtRank", 0)
                    gap = overlay.get("gapRatio")
                    nearest = overlay.get("nearestInflectionRank")
                    dist = overlay.get("distanceToInflection")
                    gap_str = f"gap={gap:.4f}" if gap else "gap=N/A"
                    near_str = f"nearest={nearest}(d={dist})" if nearest is not None else "no inflection"
                    marker = ""
                    if dist is not None and dist == 0:
                        marker = " <<< EXACT"
                    elif dist is not None and dist <= 3:
                        marker = " <-- close"
                    print(f"      rank {rank_key} ({overlay['domain']}): "
                          f"E={energy:.4f}  {gap_str}  {near_str}{marker}")

    # Cross-model comparison
    print(f"\n{'=' * 90}")
    print("ACTIVATION GAP RATIOS AT DOMAIN RANKS (cross-model)")
    print(f"{'=' * 90}")

    for rank, domain in DOMAIN_RANKS.items():
        print(f"\n  Rank {rank} ({domain}):")
        for model_tag, model_data in results.items():
            for gname, group_data in model_data.get("groups", {}).items():
                for layer_idx in TARGET_LAYERS:
                    ld = group_data.get("layers", {}).get(str(layer_idx))
                    if ld is None:
                        continue
                    overlay = ld["domainRankOverlay"].get(str(rank), {})
                    if not overlay.get("withinSpectrum"):
                        continue
                    gap = overlay.get("gapRatio")
                    energy = overlay.get("energyAtRank", 0)
                    dist = overlay.get("distanceToInflection")
                    gap_str = f"gap={gap:.4f}" if gap else "gap=N/A"
                    marker = ""
                    if gap and gap > 1.05:
                        marker = " *** GAP"
                    elif gap and gap > 1.02:
                        marker = " * notable"
                    print(f"    {model_tag:>4} L{layer_idx} {gname:30s} "
                          f"E={energy:.4f}  {gap_str}  d_inflect={dist}{marker}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation spectral analysis")
    parser.add_argument("--output", type=str, default="data/experiments")
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()), default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probes and group
    from modelcypher.core.domain.atlas.probe_loader import load_all_probes
    probes = load_all_probes()
    logger.info("Loaded %d probes", len(probes))

    probe_groups = _group_probes(probes)

    model_tags = args.models or list(MODELS.keys())

    results: dict[str, dict] = {}
    for tag in model_tags:
        result = analyze_model(MODELS[tag], tag, probe_groups)
        if result is not None:
            results[tag] = result
            tag_lower = tag.lower().replace(".", "p")
            out_path = output_dir / f"activation_spectral_{tag_lower}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("  Saved: %s", out_path)

    if not results:
        logger.error("No models analyzed")
        sys.exit(1)

    print_summary(results)


if __name__ == "__main__":
    main()
