#!/usr/bin/env python3
"""Experiment 14: Manifold Mapping.

Previous experiments asked: "Do FRBs look like semantic embeddings?"
This experiment asks: "What transformation maps FRB geometry to semantic space?"

The premise: If the shape of knowledge is invariant, FRBs are points on that
manifold - we just haven't found the coordinate transformation yet.

Method:
1. Use GramAligner to find F = pinv(FRB) @ semantic
2. Project FRBs into semantic space
3. Find nearest semantic neighbors for each FRB
4. Test: Do physically similar FRBs map to semantically similar regions?

Usage:
    poetry run python experiments/astronomy/exp14_manifold_mapping.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import GramAligner

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def get_semantic_concepts():
    """Get semantic concepts spanning different dimensions of meaning."""
    return {
        # Distance concepts
        "distance": [
            "near", "close", "nearby", "local", "proximate",
            "far", "distant", "remote", "cosmic", "extragalactic",
        ],
        # Intensity concepts
        "intensity": [
            "dim", "faint", "weak", "quiet", "subtle",
            "bright", "intense", "powerful", "loud", "brilliant",
        ],
        # Frequency/color concepts
        "frequency": [
            "low", "bass", "deep", "rumble", "infra",
            "high", "treble", "sharp", "piercing", "ultra",
        ],
        # Time concepts
        "time": [
            "brief", "flash", "instant", "momentary", "pulse",
            "sustained", "prolonged", "continuous", "persistent", "steady",
        ],
        # Energy concepts
        "energy": [
            "gentle", "soft", "mild", "calm", "passive",
            "explosive", "violent", "catastrophic", "cataclysmic", "extreme",
        ],
        # Structure concepts
        "structure": [
            "simple", "clean", "pure", "uniform", "smooth",
            "complex", "intricate", "scattered", "chaotic", "turbulent",
        ],
    }


def extract_concept_embeddings(concepts: dict, backend):
    """Extract Whisper decoder embeddings for semantic concepts."""
    from transformers import WhisperModel, WhisperProcessor
    import torch

    model = WhisperModel.from_pretrained("openai/whisper-base")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    tokenizer = processor.tokenizer

    all_embeds = []
    all_labels = []
    category_indices = {}

    start_idx = 0
    for category, words in concepts.items():
        category_indices[category] = list(range(start_idx, start_idx + len(words)))
        start_idx += len(words)

        for word in words:
            tokens = tokenizer(word, return_tensors="pt").input_ids
            with torch.no_grad():
                embed_layer = model.decoder.embed_tokens
                embeds = embed_layer(tokens)
                pooled = embeds.mean(dim=1)
                all_embeds.append(pooled)
                all_labels.append((category, word))

    embeddings = torch.cat(all_embeds, dim=0)
    return backend.array(embeddings.detach().cpu().numpy()), all_labels, category_indices


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 14: Manifold Mapping")
    print("=" * 60)
    print("\nHypothesis: FRBs are points on the information manifold.")
    print("Method: Find F that maps FRB features → semantic space.")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])
    names = [w.metadata.tns_name for w in waterfalls]

    # Extract FRB features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    # Get semantic concepts
    print("\nExtracting semantic concept embeddings...")
    concepts = get_semantic_concepts()
    concept_emb, concept_labels, category_indices = extract_concept_embeddings(concepts, backend)
    concept_np = np.array(backend.tolist(concept_emb))

    print(f"  {len(concept_labels)} concepts across {len(concepts)} categories")
    print(f"  Concept embedding dim: {concept_np.shape[1]}")
    print(f"  FRB feature dim: {frb_np.shape[1]}")

    print("\n" + "=" * 40)
    print("PART 1: GRAM ALIGNMENT")
    print("=" * 40)

    # Use GramAligner to find transformation
    aligner = GramAligner(backend)

    # We need matched samples for alignment
    # Strategy: Use FRBs as source, concepts as target
    # Subsample to match sizes
    n_align = min(n_frbs, len(concept_labels))

    # Select diverse subset of concepts
    concept_subset_idx = np.linspace(0, len(concept_labels) - 1, n_align, dtype=int)
    concept_subset = concept_np[concept_subset_idx]

    # Use all FRBs (or subset if more than concepts)
    frb_subset_idx = np.linspace(0, n_frbs - 1, n_align, dtype=int)
    frb_subset = frb_np[frb_subset_idx]

    print(f"\nAligning {n_align} FRBs to {n_align} concepts...")

    # Compute alignment
    alignment_result = aligner.align(
        backend.array(frb_subset),
        backend.array(concept_subset),
    )

    print(f"  Raw CKA (before alignment): {alignment_result.raw_cka:.4f}")
    print(f"  Aligned CKA: {alignment_result.aligned_cka:.4f}")
    print(f"  Gram condition number: {alignment_result.gram_condition_number:.2e}")

    # Get the transformation matrix
    F = np.array(backend.tolist(alignment_result.transform))
    print(f"  Transform shape: {F.shape}")

    print("\n" + "=" * 40)
    print("PART 2: PROJECT FRBs INTO SEMANTIC SPACE")
    print("=" * 40)

    # Project ALL FRBs using the learned transformation
    frb_projected = frb_np @ F

    print(f"\nProjected FRB shape: {frb_projected.shape}")
    print(f"Concept shape: {concept_np.shape}")

    # Find nearest concept for each FRB
    distances = cdist(frb_projected, concept_np, metric='cosine')
    nearest_concept_idx = np.argmin(distances, axis=1)
    nearest_distances = np.min(distances, axis=1)

    print("\n" + "=" * 40)
    print("PART 3: FRB → CONCEPT MAPPING")
    print("=" * 40)

    print("\nNearest semantic concept for each FRB:")
    frb_concept_mapping = []
    for i in range(n_frbs):
        cat, word = concept_labels[nearest_concept_idx[i]]
        frb_concept_mapping.append({
            "frb": names[i],
            "dm": float(dms[i]),
            "snr": float(snrs[i]),
            "nearest_concept": word,
            "category": cat,
            "distance": float(nearest_distances[i]),
        })
        if i < 10:  # Show first 10
            print(f"  {names[i]} (DM={dms[i]:.0f}, SNR={snrs[i]:.1f}) → '{word}' ({cat})")

    # Analyze mapping by category
    print("\n" + "=" * 40)
    print("PART 4: CATEGORY DISTRIBUTION")
    print("=" * 40)

    category_counts = {}
    for mapping in frb_concept_mapping:
        cat = mapping["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nFRBs mapped to each semantic category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} FRBs ({count/n_frbs*100:.0f}%)")

    print("\n" + "=" * 40)
    print("PART 5: PHYSICAL CORRELATIONS")
    print("=" * 40)

    # Do FRBs with similar DM map to similar concepts?
    # Group FRBs by DM quartile
    dm_quartiles = np.percentile(dms, [25, 50, 75])
    dm_groups = np.digitize(dms, dm_quartiles)

    print("\nConcept distribution by DM quartile:")
    for q in range(4):
        mask = dm_groups == q
        q_concepts = [frb_concept_mapping[i]["category"] for i in range(n_frbs) if mask[i]]
        q_counts = {}
        for c in q_concepts:
            q_counts[c] = q_counts.get(c, 0) + 1
        top_cat = max(q_counts, key=q_counts.get) if q_counts else "N/A"
        dm_range = f"Q{q+1}"
        print(f"  {dm_range}: Top category = {top_cat} ({q_counts.get(top_cat, 0)}/{sum(q_counts.values())})")

    # Test: Do nearby FRBs (low DM) map to "near" concepts?
    # Do distant FRBs (high DM) map to "far" concepts?
    distance_words = ["near", "close", "nearby", "local", "far", "distant", "remote", "cosmic"]
    near_words = ["near", "close", "nearby", "local", "proximate"]
    far_words = ["far", "distant", "remote", "cosmic", "extragalactic"]

    # Find FRBs that mapped to distance-related concepts
    near_frb_dms = [m["dm"] for m in frb_concept_mapping if m["nearest_concept"] in near_words]
    far_frb_dms = [m["dm"] for m in frb_concept_mapping if m["nearest_concept"] in far_words]

    print("\n" + "=" * 40)
    print("PART 6: SEMANTIC COHERENCE TEST")
    print("=" * 40)

    print("\nDo low-DM FRBs map to 'near' concepts?")
    print(f"  FRBs mapped to 'near' concepts: {len(near_frb_dms)}")
    if near_frb_dms:
        print(f"    Mean DM: {np.mean(near_frb_dms):.0f}")
    print(f"  FRBs mapped to 'far' concepts: {len(far_frb_dms)}")
    if far_frb_dms:
        print(f"    Mean DM: {np.mean(far_frb_dms):.0f}")

    # Statistical test
    if len(near_frb_dms) >= 2 and len(far_frb_dms) >= 2:
        t_stat, p_val = stats.ttest_ind(near_frb_dms, far_frb_dms)
        print(f"\n  t-test (near vs far DM): t={t_stat:.2f}, p={p_val:.3f}")
        if p_val < 0.05 and np.mean(near_frb_dms) < np.mean(far_frb_dms):
            print("  ** LOW-DM FRBs map to 'near', HIGH-DM to 'far' **")
        elif p_val < 0.05:
            print("  ** Significant but OPPOSITE direction **")
        else:
            print("  No significant difference")

    # Test intensity mapping
    dim_words = ["dim", "faint", "weak", "quiet", "subtle"]
    bright_words = ["bright", "intense", "powerful", "loud", "brilliant"]

    dim_frb_snrs = [m["snr"] for m in frb_concept_mapping if m["nearest_concept"] in dim_words]
    bright_frb_snrs = [m["snr"] for m in frb_concept_mapping if m["nearest_concept"] in bright_words]

    print("\nDo low-SNR FRBs map to 'dim' concepts?")
    print(f"  FRBs mapped to 'dim' concepts: {len(dim_frb_snrs)}")
    if dim_frb_snrs:
        print(f"    Mean SNR: {np.mean(dim_frb_snrs):.1f}")
    print(f"  FRBs mapped to 'bright' concepts: {len(bright_frb_snrs)}")
    if bright_frb_snrs:
        print(f"    Mean SNR: {np.mean(bright_frb_snrs):.1f}")

    if len(dim_frb_snrs) >= 2 and len(bright_frb_snrs) >= 2:
        t_stat, p_val = stats.ttest_ind(dim_frb_snrs, bright_frb_snrs)
        print(f"\n  t-test (dim vs bright SNR): t={t_stat:.2f}, p={p_val:.3f}")
        if p_val < 0.05 and np.mean(dim_frb_snrs) < np.mean(bright_frb_snrs):
            print("  ** LOW-SNR FRBs map to 'dim', HIGH-SNR to 'bright' **")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print(f"\nAlignment quality: CKA {alignment_result.raw_cka:.2f} → {alignment_result.aligned_cka:.2f}")

    if alignment_result.aligned_cka > 0.8:
        print("** STRONG alignment found **")
        print("→ FRB feature geometry can be mapped to semantic geometry")
    elif alignment_result.aligned_cka > 0.5:
        print("** MODERATE alignment **")
        print("→ Partial geometric correspondence exists")
    else:
        print("** WEAK alignment **")
        print("→ FRB geometry differs substantially from semantic geometry")

    results = {
        "experiment": "exp14_manifold_mapping",
        "timestamp": datetime.now().isoformat(),
        "n_frbs": n_frbs,
        "n_concepts": len(concept_labels),
        "alignment": {
            "raw_cka": float(alignment_result.raw_cka),
            "aligned_cka": float(alignment_result.aligned_cka),
            "gram_condition_number": float(alignment_result.gram_condition_number),
        },
        "category_distribution": category_counts,
        "frb_concept_mapping": frb_concept_mapping,
    }

    output_path = results_dir / "exp14_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
