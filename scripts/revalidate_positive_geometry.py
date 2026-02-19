#!/usr/bin/env python3
"""Re-validate positive geometry signatures with corrected LFM2 mask routing.

The original positive geometry analysis (via deleted `mc analyze concept-volume`)
used forward_through_backbone() which applied a numeric causal mask to ALL layers.
LFM2 hybrid models need string "causal" for attention layers and None for conv layers.
That bug is now fixed in _resolve_layer_mask().

This script re-runs the same analysis with corrected mask routing and outputs JSON
in the same schema (mc.geometry.research.positive_geometry.v1) for direct comparison.

Usage:
    poetry run python scripts/revalidate_positive_geometry.py --models 350M
    poetry run python scripts/revalidate_positive_geometry.py
"""

from __future__ import annotations

import argparse
import hashlib
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

MODELS = {
    "350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
}

# Domain groups matching the original positive geometry analysis
DOMAIN_GROUPS = {
    "math": ["mathematical", "logical"],
    "linguistic": ["linguistic", "mental"],
    "computational": ["computational", "structural"],
    "affective": ["affective", "relational"],
    "temporal_spatial": ["temporal", "spatial"],
    "moral": ["moral"],
    "safety": ["safety"],
    "philosophical": ["philosophical"],
    "physical": ["physical"],
    "factual": ["factual"],
}

TARGET_LAYERS = [7, 8]
PROBE_COUNT_CAP = 256
MAX_MINORS = 256
RANK_SOURCE = "spectral-gap"
SELECTION = "lexicographic"


def _group_probes(probes: list) -> dict[str, list]:
    """Group probes by domain group. Only probes with support_texts included."""
    from modelcypher.core.domain.domains import AtlasDomain

    groups: dict[str, list] = {}
    for gname, domain_values in DOMAIN_GROUPS.items():
        target_domains = {AtlasDomain(v) for v in domain_values}
        gprobes = [p for p in probes if p.domain in target_domains and p.support_texts]
        # Cap at PROBE_COUNT_CAP
        if len(gprobes) > PROBE_COUNT_CAP:
            gprobes = gprobes[:PROBE_COUNT_CAP]
        groups[gname] = gprobes
        logger.info("  Group %s: %d probes", gname, len(gprobes))

    return groups


def _forward_to_layer(embed_tokens, layers, input_ids, target_layer: int, backend):
    """Forward pass with LFM2-compatible mask routing."""
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
):
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

        # Mean-pool over sequence dim
        h = backend.mean(hidden, axis=1)
        backend.eval(h)
        vecs.append(h)

        if (i + 1) % 50 == 0:
            logger.info("      %d/%d probes collected", i + 1, len(probes))

    stacked = backend.concatenate(vecs, axis=0)
    backend.eval(stacked)
    # Cast to float32 — MLX SVD requires float32 (model outputs bf16)
    stacked = backend.astype(stacked, "float32")
    backend.eval(stacked)
    return stacked


def _probe_order_hash(probes: list) -> str:
    """Compute hash of probe ordering for reproducibility tracking."""
    ids = [p.id for p in probes]
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


def analyze_model(
    model_path: str,
    model_tag: str,
    probe_groups: dict[str, list],
    output_dir: Path,
) -> dict[str, dict]:
    """Run positive geometry analysis for one model, all domain groups."""
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s", model_path)
        return {}

    logger.info("Loading model: %s", model_tag)

    from modelcypher.backends import initialize_default_backend
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.adapters.model_backbone import resolve_model_backbone
    from modelcypher.core.domain.geometry.positive_geometry import (
        compute_positive_grassmann_signature,
    )

    backend = initialize_default_backend()
    model, tokenizer = load_model_for_training(model_path)
    backbone = resolve_model_backbone(model)

    if backbone is None:
        logger.error("Cannot resolve backbone for %s", model_path)
        return {}

    results: dict[str, dict] = {}

    for gname, probes in probe_groups.items():
        if not probes:
            logger.info("  Skipping %s: no probes", gname)
            continue

        logger.info("  Group %s (%d probes)", gname, len(probes))

        probe_hash = _probe_order_hash(probes)
        domains = DOMAIN_GROUPS[gname]

        sweep_reports: list[dict] = []

        layer_reports: list[dict] = []
        for target_layer in TARGET_LAYERS:
            logger.info("    Layer %d: collecting activations...", target_layer)

            acts = _collect_activations(
                model, tokenizer, backbone, backend, probes, target_layer,
            )

            logger.info("    Layer %d: computing signature (%s)...",
                        target_layer, list(acts.shape))

            sig = compute_positive_grassmann_signature(
                activations=acts,
                backend=backend,
                max_minors=MAX_MINORS,
                selection=SELECTION,
                rank_source=RANK_SOURCE,
            )

            layer_reports.append({
                "layer": target_layer,
                "activationCount": int(acts.shape[0]),
                "signature": sig.to_dict(),
            })

            logger.info("    Layer %d: rank=%d posFrac=%.4f signEnt=%.4f zeroFrac=%.4f",
                        target_layer, sig.subspace_rank,
                        sig.positive_fraction, sig.sign_entropy, sig.zero_fraction)

        sweep_reports.append({
            "shuffleIndex": 0,
            "probeOrderHash": probe_hash,
            "layerReports": layer_reports,
        })

        # Build output in original schema
        output_data = {
            "_schema": "mc.geometry.research.positive_geometry.v1",
            "_note": "Revalidated with corrected LFM2 mask routing (2026-02-19)",
            "modelPath": model_path,
            "probeCount": len(probes),
            "domains": domains,
            "maxMinors": MAX_MINORS,
            "selection": SELECTION,
            "rankSource": RANK_SOURCE,
            "layers": TARGET_LAYERS,
            "shuffleSeed": None,
            "shuffleCount": 1,
            "sweeps": sweep_reports,
        }

        # Save per domain group
        tag_lower = model_tag.lower().replace(".", "p")
        out_path = output_dir / f"revalidated_positive_geometry_{tag_lower}_domains_{gname}.json"
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("    Saved: %s", out_path)

        results[gname] = output_data

    return results


def print_comparison(all_results: dict[str, dict[str, dict]], original_dir: Path | None) -> None:
    """Print comparison table matching the format in the research note."""
    print()
    print("=" * 100)
    print("REVALIDATED POSITIVE GEOMETRY — CORRECTED LFM2 MASK ROUTING")
    print("=" * 100)
    print()
    print("Format per cell: rank, posFraction, signEntropy, zeroFraction")
    print()

    # Build table rows
    for gname in DOMAIN_GROUPS:
        for layer in TARGET_LAYERS:
            cells = []
            for model_tag in ["350M", "700M", "1.2B"]:
                model_data = all_results.get(model_tag, {}).get(gname)
                if model_data is None:
                    cells.append("N/A")
                    continue

                # Find the layer report
                sig_data = None
                for sweep in model_data.get("sweeps", []):
                    for lr in sweep.get("layerReports", []):
                        if lr["layer"] == layer:
                            sig_data = lr["signature"]
                            break

                if sig_data is None:
                    cells.append("N/A")
                    continue

                rank = sig_data["subspaceRank"]
                pos = sig_data["positiveFraction"]
                ent = sig_data["signEntropy"]
                zero = sig_data["zeroFraction"]
                cells.append(f"{rank},{pos:.4f},{ent:.4f},{zero:.4f}")

            abbrev = gname[:8].ljust(8)
            print(f"{abbrev} L{layer}  {cells[0]:>35s}  {cells[1]:>35s}  {cells[2]:>35s}")

    print()

    # If we have original data, show diffs
    if original_dir and original_dir.exists():
        print("=" * 100)
        print("DELTA FROM ORIGINAL (buggy mask) → REVALIDATED (corrected mask)")
        print("=" * 100)
        print()

        for model_tag in ["350M", "700M", "1.2B"]:
            tag_lower = model_tag.lower().replace(".", "p")

            for gname in DOMAIN_GROUPS:
                # Load original
                orig_path = original_dir / f"positive_geometry_lfm2_{tag_lower}_domains_{gname}.json"
                if not orig_path.exists():
                    continue

                with open(orig_path) as f:
                    orig = json.load(f)

                new_data = all_results.get(model_tag, {}).get(gname)
                if new_data is None:
                    continue

                for layer in TARGET_LAYERS:
                    # Find original signature
                    orig_sig = None
                    for sweep in orig.get("sweeps", []):
                        for lr in sweep.get("layerReports", []):
                            if lr["layer"] == layer:
                                orig_sig = lr["signature"]
                    if orig_sig is None:
                        # Try flat layerReports (some original files use this)
                        for lr in orig.get("layerReports", []):
                            if lr["layer"] == layer:
                                orig_sig = lr["signature"]

                    new_sig = None
                    for sweep in new_data.get("sweeps", []):
                        for lr in sweep.get("layerReports", []):
                            if lr["layer"] == layer:
                                new_sig = lr["signature"]

                    if orig_sig is None or new_sig is None:
                        continue

                    # Compare key metrics
                    diffs = []
                    for key, label in [
                        ("subspaceRank", "rank"),
                        ("positiveFraction", "posFrac"),
                        ("signEntropy", "signEnt"),
                        ("zeroFraction", "zeroFrac"),
                    ]:
                        old_val = orig_sig.get(key, orig_sig.get(
                            # Handle camelCase variations from original format
                            key[0].lower() + key[1:], 0
                        ))
                        new_val = new_sig.get(key, 0)
                        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                            delta = new_val - old_val
                            if abs(delta) > 1e-6:
                                diffs.append(f"{label}: {old_val:.4f}→{new_val:.4f} (Δ{delta:+.4f})")

                    if diffs:
                        print(f"  {model_tag} L{layer} {gname}: {', '.join(diffs)}")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-validate positive geometry with corrected masks")
    parser.add_argument("--output", type=str, default="data/experiments")
    parser.add_argument("--models", nargs="*", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--original-dir", type=str, default=None,
                        help="Directory containing original (buggy) data for comparison")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probes and group
    from modelcypher.core.domain.atlas.probe_loader import load_all_probes
    probes = load_all_probes()
    logger.info("Loaded %d probes", len(probes))

    probe_groups = _group_probes(probes)

    model_tags = args.models or list(MODELS.keys())

    all_results: dict[str, dict[str, dict]] = {}
    for tag in model_tags:
        results = analyze_model(MODELS[tag], tag, probe_groups, output_dir)
        if results:
            all_results[tag] = results

    if not all_results:
        logger.error("No models analyzed")
        sys.exit(1)

    original_dir = Path(args.original_dir) if args.original_dir else None
    print_comparison(all_results, original_dir)


if __name__ == "__main__":
    main()
