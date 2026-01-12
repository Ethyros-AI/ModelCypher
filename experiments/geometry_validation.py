#!/usr/bin/env python3
"""
Geometry Validation Experiment

Generates concrete experimental data to support geometric claims:
1. Intrinsic dimension of semantic prime subspace
2. Within-category vs between-category similarity
3. CKA coherence across concept domains
4. Mean pairwise similarity by domain

Results saved to experiments/results/geometry_validation.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run geometry validation experiments."""
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
    from modelcypher.core.domain.agents.unified_atlas import (
        AtlasSource,
        UnifiedAtlasInventory,
    )
    from modelcypher.core.domain.geometry.cka import compute_cka
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_matrix
    from modelcypher.core.use_cases.atlas_bootstrap import (
        register_default_atlas_inventories,
    )

    register_default_atlas_inventories()
    backend = get_default_backend()

    # Find available model
    model_paths = [
        Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots",
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots",
    ]

    model_path = None
    for p in model_paths:
        if p.exists():
            snapshots = list(p.iterdir())
            if snapshots:
                model_path = snapshots[0]
                break

    if model_path is None:
        logger.error("No models found. Please download SmolLM-135M first:")
        logger.error("  poetry run python -c \"from transformers import AutoModel; AutoModel.from_pretrained('HuggingFaceTB/SmolLM-135M')\"")
        sys.exit(1)

    logger.info(f"Using model: {model_path}")

    # Load model and tokenizer
    model, tokenizer = load_model_for_training(str(model_path))
    backbone = resolve_model_backbone(model)
    if backbone is None:
        logger.error("Failed to resolve model backbone")
        sys.exit(1)

    embed_tokens, layers, norm = backbone
    num_layers = len(layers)
    logger.info(f"Model has {num_layers} layers")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": str(model_path.name),
        "num_layers": num_layers,
        "experiments": {},
    }

    # =========================================================================
    # Experiment 1: Semantic Primes Geometry
    # =========================================================================
    logger.info("\n=== Experiment 1: Semantic Primes Geometry ===")

    primes = SemanticPrimeInventory.english2014()
    logger.info(f"Loaded {len(primes)} semantic primes")

    class PrimeAnchor:
        def __init__(self, prime):
            self.name = prime.id
            self.prompt = prime.canonical_english
            self.category = prime.category.value

    anchors = [PrimeAnchor(p) for p in primes]

    # Extract at middle layer
    mid_layer = num_layers // 2
    activations = extract_anchor_activations(
        anchors=anchors,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        layers=layers,
        norm=norm,
        target_layer=mid_layer,
        backend=backend,
        prompt_attr="prompt",
        name_attr="name",
    )

    logger.info(f"Extracted {len(activations)} prime activations at layer {mid_layer}")

    # Stack into matrix
    prime_ids = list(activations.keys())
    vectors = [activations[pid] for pid in prime_ids]
    matrix = backend.stack(vectors, axis=0)
    backend.eval(matrix)

    # Compute intrinsic dimension
    id_computer = IntrinsicDimension(backend)
    try:
        id_result = id_computer.compute(matrix)
        intrinsic_dim = float(id_result.intrinsic_dimension)
        if id_result.ci is not None:
            id_ci = (float(id_result.ci.lower), float(id_result.ci.upper))
            logger.info(
                "Intrinsic dimension: %.2f (95%% CI: [%.2f, %.2f])",
                intrinsic_dim,
                id_ci[0],
                id_ci[1],
            )
        else:
            id_ci = (None, None)
            logger.info("Intrinsic dimension: %.2f (CI unavailable)", intrinsic_dim)
    except Exception as e:
        logger.warning("Intrinsic dimension computation failed: %s", e)
        intrinsic_dim = None
        id_ci = (None, None)

    # Compute pairwise similarities
    cos_matrix = geodesic_cosine_matrix(matrix, backend)
    backend.eval(cos_matrix)

    # Calculate within-category vs between-category similarity
    prime_lookup = {p.id: p for p in primes}
    within_cat_sims = []
    between_cat_sims = []

    n = len(prime_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(backend.to_scalar(cos_matrix[i, j]))
            prime_i = prime_lookup.get(prime_ids[i])
            prime_j = prime_lookup.get(prime_ids[j])
            if prime_i and prime_j:
                if prime_i.category == prime_j.category:
                    within_cat_sims.append(sim)
                else:
                    between_cat_sims.append(sim)

    within_mean = sum(within_cat_sims) / len(within_cat_sims) if within_cat_sims else 0
    between_mean = sum(between_cat_sims) / len(between_cat_sims) if between_cat_sims else 0
    separation_ratio = within_mean / between_mean if between_mean > 0 else float('inf')

    logger.info(f"Within-category similarity: {within_mean:.4f}")
    logger.info(f"Between-category similarity: {between_mean:.4f}")
    logger.info(f"Separation ratio: {separation_ratio:.4f}")

    results["experiments"]["semantic_primes"] = {
        "count": len(activations),
        "layer": mid_layer,
        "intrinsic_dimension": {
            "value": intrinsic_dim,
            "ci_lower": id_ci[0],
            "ci_upper": id_ci[1],
        },
        "within_category_similarity": within_mean,
        "between_category_similarity": between_mean,
        "separation_ratio": separation_ratio,
        "categories_cluster": separation_ratio > 1.0,
    }

    # =========================================================================
    # Experiment 2: Cross-Domain CKA
    # =========================================================================
    logger.info("\n=== Experiment 2: Cross-Domain CKA ===")

    all_probes = UnifiedAtlasInventory.all_probes()
    domains = {
        "spatial": AtlasSource.SPATIAL_CONCEPT,
        "temporal": AtlasSource.TEMPORAL_CONCEPT,
        "social": AtlasSource.SOCIAL_CONCEPT,
        "moral": AtlasSource.MORAL_CONCEPT,
    }

    domain_activations = {}
    for domain_name, source in domains.items():
        probes = [p for p in all_probes if p.source == source][:20]  # Limit to 20 per domain
        if not probes:
            continue

        class ProbeAnchor:
            def __init__(self, probe):
                self.name = probe.name
                self.prompt = probe.support_texts[0] if probe.support_texts else probe.name

        anchors = [ProbeAnchor(p) for p in probes]
        acts = extract_anchor_activations(
            anchors=anchors,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            layers=layers,
            norm=norm,
            target_layer=mid_layer,
            backend=backend,
            prompt_attr="prompt",
            name_attr="name",
        )
        if len(acts) >= 5:
            names = list(acts.keys())
            vecs = [acts[n] for n in names]
            mat = backend.stack(vecs, axis=0)
            backend.eval(mat)
            domain_activations[domain_name] = mat
            logger.info(f"  {domain_name}: {len(acts)} probes extracted")

    # Compute CKA between domains
    cka_results = {}
    domain_names = list(domain_activations.keys())
    for i, d1 in enumerate(domain_names):
        for d2 in domain_names[i + 1:]:
            # Need same number of samples for CKA
            m1 = domain_activations[d1]
            m2 = domain_activations[d2]
            n1, n2 = backend.shape(m1)[0], backend.shape(m2)[0]
            min_n = min(n1, n2)
            m1_trim = m1[:min_n]
            m2_trim = m2[:min_n]

            cka_result = compute_cka(m1_trim, m2_trim, backend)
            cka_val = float(cka_result.cka)
            cka_results[f"{d1}_vs_{d2}"] = cka_val
            logger.info(f"  CKA({d1}, {d2}): {cka_val:.4f}")

    results["experiments"]["cross_domain_cka"] = {
        "layer": mid_layer,
        "domains": domain_names,
        "cka_pairs": cka_results,
        "mean_cka": sum(cka_results.values()) / len(cka_results) if cka_results else 0,
    }

    # =========================================================================
    # Experiment 3: Alignment Invariance (Raw vs Aligned CKA)
    # =========================================================================
    logger.info("\n=== Experiment 3: Alignment Invariance ===")
    logger.info("Demonstrating that structure is invariant after Procrustes alignment")

    # Extract same probes from two different layers
    early_layer = num_layers // 4
    late_layer = 3 * num_layers // 4

    # Use spatial probes as test case
    spatial_probes = [p for p in all_probes if p.source == AtlasSource.SPATIAL_CONCEPT][:15]

    class ProbeAnchor:
        def __init__(self, probe):
            self.name = probe.name
            self.prompt = probe.support_texts[0] if probe.support_texts else probe.name

    spatial_anchors = [ProbeAnchor(p) for p in spatial_probes]

    # Extract from early layer
    early_acts = extract_anchor_activations(
        anchors=spatial_anchors,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        layers=layers,
        norm=norm,
        target_layer=early_layer,
        backend=backend,
        prompt_attr="prompt",
        name_attr="name",
    )

    # Extract from late layer
    late_acts = extract_anchor_activations(
        anchors=spatial_anchors,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        layers=layers,
        norm=norm,
        target_layer=late_layer,
        backend=backend,
        prompt_attr="prompt",
        name_attr="name",
    )

    alignment_results = {}
    if len(early_acts) >= 5 and len(late_acts) >= 5:
        # Stack activations (same probe order)
        common_names = [n for n in early_acts.keys() if n in late_acts]
        early_mat = backend.stack([early_acts[n] for n in common_names], axis=0)
        late_mat = backend.stack([late_acts[n] for n in common_names], axis=0)
        backend.eval(early_mat)
        backend.eval(late_mat)

        # Raw CKA (before alignment) - different coordinate systems
        raw_cka_result = compute_cka(early_mat, late_mat, backend)
        raw_cka = float(raw_cka_result.cka)
        logger.info(f"  Raw CKA (layer {early_layer} vs {late_layer}): {raw_cka:.4f}")

        # Find alignment transform
        alignment = find_alignment(early_mat, late_mat, backend)

        # Apply alignment
        aligned_early = backend.matmul(early_mat, alignment.feature_transform)
        backend.eval(aligned_early)

        # CKA after alignment - invariant structure revealed
        aligned_cka_result = compute_cka(aligned_early, late_mat, backend)
        aligned_cka = float(aligned_cka_result.cka)
        logger.info(f"  Aligned CKA (after Procrustes): {aligned_cka:.4f}")
        logger.info(f"  Alignment achieved_cka: {alignment.achieved_cka:.4f}")
        logger.info(f"  Is perfect alignment: {alignment.is_perfect}")

        alignment_results = {
            "early_layer": early_layer,
            "late_layer": late_layer,
            "probe_count": len(common_names),
            "raw_cka": raw_cka,
            "aligned_cka": aligned_cka,
            "alignment_achieved_cka": alignment.achieved_cka,
            "is_perfect": alignment.is_perfect,
            "numerical_deviation": alignment.numerical_deviation,
            "precision_threshold": alignment.precision_threshold,
        }

    results["experiments"]["alignment_invariance"] = alignment_results

    # =========================================================================
    # Experiment 4: Layer-wise Intrinsic Dimension
    # =========================================================================
    logger.info("\n=== Experiment 4: Layer-wise Intrinsic Dimension ===")

    layer_ids = []
    test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]

    for layer_idx in test_layers:
        acts = extract_anchor_activations(
            anchors=[PrimeAnchor(p) for p in primes[:30]],  # Use first 30 primes
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            layers=layers,
            norm=norm,
            target_layer=layer_idx,
            backend=backend,
            prompt_attr="prompt",
            name_attr="name",
        )
        if len(acts) >= 10:
            names = list(acts.keys())
            vecs = [acts[n] for n in names]
            mat = backend.stack(vecs, axis=0)
            backend.eval(mat)

            try:
                id_result = id_computer.compute(mat)
                layer_id = float(id_result.intrinsic_dimension)
                logger.info(f"  Layer {layer_idx}: ID = {layer_id:.2f}")
                layer_ids.append({"layer": layer_idx, "intrinsic_dimension": layer_id})
            except Exception:
                logger.warning(f"  Layer {layer_idx}: ID computation failed")

    results["experiments"]["layerwise_intrinsic_dimension"] = {
        "layers": layer_ids,
        "pattern": "ID typically increases then decreases through layers" if layer_ids else "insufficient data",
    }

    # =========================================================================
    # Experiment 5: Domain Geometry Summary
    # =========================================================================
    logger.info("\n=== Experiment 5: Domain Geometry Summary ===")

    domain_summary = {}
    for domain_name, source in domains.items():
        probes = [p for p in all_probes if p.source == source]
        if not probes:
            continue

        class ProbeAnchor:
            def __init__(self, probe):
                self.name = probe.name
                self.prompt = probe.support_texts[0] if probe.support_texts else probe.name

        anchors = [ProbeAnchor(p) for p in probes]
        acts = extract_anchor_activations(
            anchors=anchors,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            layers=layers,
            norm=norm,
            target_layer=mid_layer,
            backend=backend,
            prompt_attr="prompt",
            name_attr="name",
        )

        if len(acts) >= 5:
            names = list(acts.keys())
            vecs = [acts[n] for n in names]
            mat = backend.stack(vecs, axis=0)
            backend.eval(mat)

            # Mean pairwise similarity
            cos_mat = geodesic_cosine_matrix(mat, backend)
            backend.eval(cos_mat)

            total_sim = 0.0
            count = 0
            n = len(names)
            for i in range(n):
                for j in range(i + 1, n):
                    total_sim += float(backend.to_scalar(cos_mat[i, j]))
                    count += 1

            mean_sim = total_sim / count if count > 0 else 0

            # Intrinsic dimension
            try:
                id_result = id_computer.compute(mat)
                domain_id = float(id_result.intrinsic_dimension)
            except Exception:
                domain_id = None

            domain_summary[domain_name] = {
                "probe_count": len(acts),
                "mean_pairwise_similarity": mean_sim,
                "intrinsic_dimension": domain_id,
            }
            id_str = f"{domain_id:.2f}" if domain_id is not None else "N/A"
            logger.info(f"  {domain_name}: {len(acts)} probes, mean_sim={mean_sim:.4f}, ID={id_str}")

    results["experiments"]["domain_geometry"] = domain_summary

    # =========================================================================
    # Save Results
    # =========================================================================
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "geometry_validation.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n=== Results saved to {output_file} ===")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Geometric Claims Validation")
    logger.info("=" * 60)

    if results["experiments"].get("semantic_primes", {}).get("categories_cluster"):
        logger.info("[VALIDATED] Semantic categories cluster in embedding space")
        logger.info(f"  - Separation ratio: {results['experiments']['semantic_primes']['separation_ratio']:.2f}x")

    cka_data = results["experiments"].get("cross_domain_cka", {})
    if cka_data.get("mean_cka", 0) > 0.5:
        logger.info("[VALIDATED] Cross-domain geometric structure is consistent")
        logger.info(f"  - Mean CKA: {cka_data['mean_cka']:.4f}")

    primes_data = results["experiments"].get("semantic_primes", {})
    if primes_data.get("intrinsic_dimension", {}).get("value"):
        logger.info("[MEASURED] Intrinsic dimension of semantic prime manifold")
        id_val = primes_data["intrinsic_dimension"]["value"]
        logger.info(f"  - ID: {id_val:.2f} (ambient dim much higher)")

    # KEY RESULT: Alignment invariance
    align_data = results["experiments"].get("alignment_invariance", {})
    if align_data:
        raw_cka = align_data.get("raw_cka", 0)
        aligned_cka = align_data.get("aligned_cka", 0)
        is_perfect = align_data.get("is_perfect", False)
        logger.info("[KEY RESULT] Alignment Invariance Demonstration")
        logger.info(f"  - Raw CKA (before alignment): {raw_cka:.4f}")
        logger.info(f"  - Aligned CKA (after Procrustes): {aligned_cka:.4f}")
        logger.info(f"  - Perfect alignment achieved: {is_perfect}")
        if aligned_cka > 0.99:
            logger.info("  => VALIDATED: Structure is invariant - coordinate systems differ, relationships preserved")

    return results


if __name__ == "__main__":
    main()
