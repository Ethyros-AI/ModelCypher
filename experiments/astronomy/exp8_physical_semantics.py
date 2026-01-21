#!/usr/bin/env python3
"""Experiment 8: Physical Semantics Grounding.

Arrival insight: The test should use the FRBs' ACTUAL physical properties
as semantic grounding, not arbitrary numbered sequences.

Key idea: If FRB features encode physical information (distance, energy),
then FRBs should align with concepts that MATCH their physical properties.

Tests:
1. FRBs sorted by DM vs "distance" concepts sorted by distance
2. FRBs sorted by SNR vs "intensity" concepts sorted by intensity
3. Does physical ordering produce higher CKA than arbitrary ordering?

If FRBs are just noise: ordering shouldn't matter
If FRBs encode physics: physics-matched ordering should give higher CKA

Usage:
    poetry run python experiments/astronomy/exp8_physical_semantics.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features, extract_frb_features


def extract_concept_embeddings(concepts: list[str], backend):
    """Extract Whisper decoder embeddings for semantic concepts."""
    from transformers import WhisperModel, WhisperProcessor
    import torch

    model = WhisperModel.from_pretrained("openai/whisper-base")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    tokenizer = processor.tokenizer

    all_embeds = []
    for concept in concepts:
        tokens = tokenizer(concept, return_tensors="pt").input_ids
        with torch.no_grad():
            embed_layer = model.decoder.embed_tokens
            embeds = embed_layer(tokens)
            pooled = embeds.mean(dim=1)
            all_embeds.append(pooled)

    embeddings = torch.cat(all_embeds, dim=0)
    return backend.array(embeddings.detach().cpu().numpy())


def generate_noise_features(n_samples: int, backend, seed: int = 42):
    """Generate features from white noise."""
    rng = np.random.default_rng(seed)
    features = []

    for i in range(n_samples):
        n_freq, n_time = 256, 1024
        waterfall = rng.standard_normal((n_freq, n_time)).astype(np.float32)
        waterfall = backend.array(waterfall)
        time_series = backend.array(rng.standard_normal(n_time).astype(np.float32))
        spectrum = backend.array(rng.standard_normal(n_freq).astype(np.float32))

        frb_feat = extract_frb_features(
            waterfall, time_series, spectrum, backend,
            tns_name=f"noise_{i}"
        )
        features.append(backend.tolist(frb_feat.features))

    return np.array(features)


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 8: Physical Semantics Grounding")
    print("=" * 60)
    print("\nHypothesis: If FRBs encode physical information,")
    print("they should align better with concepts MATCHED to their physics.")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])

    print(f"DM range: {dms.min():.1f} - {dms.max():.1f} pc/cm³")
    print(f"SNR range: {snrs.min():.1f} - {snrs.max():.1f}")

    # Extract features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    # Generate noise
    noise_np = generate_noise_features(n_frbs, backend)

    # Create PHYSICALLY-GROUNDED concepts
    # Key: The concept's semantic content MATCHES the physical property

    # Normalize DM to create distance concepts
    dm_normalized = (dms - dms.min()) / (dms.max() - dms.min())
    distance_concepts = [
        f"a distance of {int(d * 1000)} light years away"
        for d in dm_normalized
    ]

    # Normalize SNR to create intensity concepts
    snr_normalized = (snrs - snrs.min()) / (snrs.max() - snrs.min())
    intensity_concepts = [
        f"brightness level {int(s * 100)} percent"
        for s in snr_normalized
    ]

    print("\n" + "=" * 40)
    print("PART 1: PHYSICS-MATCHED ALIGNMENT")
    print("=" * 40)

    # Test 1: FRBs ordered by DM vs distance concepts ordered by distance
    print("\n--- DM-Distance Alignment ---")
    print("FRBs sorted by DM ↔ Distance concepts sorted by distance")

    dm_order = np.argsort(dms)
    frb_dm_sorted = frb_np[dm_order]

    # Distance concepts already scale with DM
    dist_emb = extract_concept_embeddings(distance_concepts, backend)
    # Sort embeddings by the same DM order
    dist_emb_sorted_np = np.array(backend.tolist(dist_emb))[dm_order]
    dist_emb_sorted = backend.array(dist_emb_sorted_np)

    dm_matched_cka = compute_cka(
        backend.array(frb_dm_sorted), dist_emb_sorted, backend=backend
    )
    print(f"  Physics-matched CKA: {dm_matched_cka.cka:.4f}")

    # Control: FRBs in random order vs same distance concepts
    rng = np.random.default_rng(42)
    random_order = rng.permutation(n_frbs)
    frb_random = frb_np[random_order]

    dm_random_cka = compute_cka(
        backend.array(frb_random), dist_emb_sorted, backend=backend
    )
    print(f"  Random-order CKA: {dm_random_cka.cka:.4f}")

    dm_improvement = dm_matched_cka.cka - dm_random_cka.cka
    print(f"  Improvement from physics-matching: {dm_improvement:+.4f}")

    # Test 2: FRBs ordered by SNR vs intensity concepts
    print("\n--- SNR-Intensity Alignment ---")
    print("FRBs sorted by SNR ↔ Intensity concepts sorted by brightness")

    snr_order = np.argsort(snrs)
    frb_snr_sorted = frb_np[snr_order]

    int_emb = extract_concept_embeddings(intensity_concepts, backend)
    int_emb_sorted_np = np.array(backend.tolist(int_emb))[snr_order]
    int_emb_sorted = backend.array(int_emb_sorted_np)

    snr_matched_cka = compute_cka(
        backend.array(frb_snr_sorted), int_emb_sorted, backend=backend
    )
    print(f"  Physics-matched CKA: {snr_matched_cka.cka:.4f}")

    snr_random_cka = compute_cka(
        backend.array(frb_random), int_emb_sorted, backend=backend
    )
    print(f"  Random-order CKA: {snr_random_cka.cka:.4f}")

    snr_improvement = snr_matched_cka.cka - snr_random_cka.cka
    print(f"  Improvement from physics-matching: {snr_improvement:+.4f}")

    print("\n" + "=" * 40)
    print("PART 2: NOISE COMPARISON")
    print("=" * 40)
    print("\nDoes noise also improve with 'physics-matching'?")
    print("(It shouldn't - noise has no physical structure)")

    # Noise with "physics-matched" ordering (meaningless for noise)
    noise_dm_matched_cka = compute_cka(
        backend.array(noise_np[dm_order]), dist_emb_sorted, backend=backend
    )
    noise_dm_random_cka = compute_cka(
        backend.array(noise_np[random_order]), dist_emb_sorted, backend=backend
    )
    noise_dm_diff = noise_dm_matched_cka.cka - noise_dm_random_cka.cka

    print(f"\nNoise DM-'matched': {noise_dm_matched_cka.cka:.4f}")
    print(f"Noise random: {noise_dm_random_cka.cka:.4f}")
    print(f"Noise 'improvement': {noise_dm_diff:+.4f}")

    print("\n" + "=" * 40)
    print("PART 3: CROSS-VALIDATION")
    print("=" * 40)
    print("\nDoes DM-sorting improve alignment with DISTANCE concepts")
    print("but NOT with INTENSITY concepts? (And vice versa)")

    # DM-sorted FRBs with intensity concepts (should NOT improve)
    dm_sorted_vs_intensity = compute_cka(
        backend.array(frb_dm_sorted), int_emb_sorted, backend=backend
    )
    random_vs_intensity = compute_cka(
        backend.array(frb_random), int_emb_sorted, backend=backend
    )
    cross_dm_int = dm_sorted_vs_intensity.cka - random_vs_intensity.cka

    print(f"\nDM-sorted vs Intensity concepts: {dm_sorted_vs_intensity.cka:.4f}")
    print(f"Random vs Intensity concepts: {random_vs_intensity.cka:.4f}")
    print(f"Cross-domain effect: {cross_dm_int:+.4f}")

    # SNR-sorted FRBs with distance concepts (should NOT improve)
    snr_sorted_vs_distance = compute_cka(
        backend.array(frb_snr_sorted), dist_emb_sorted, backend=backend
    )
    random_vs_distance = compute_cka(
        backend.array(frb_random), dist_emb_sorted, backend=backend
    )
    cross_snr_dist = snr_sorted_vs_distance.cka - random_vs_distance.cka

    print(f"\nSNR-sorted vs Distance concepts: {snr_sorted_vs_distance.cka:.4f}")
    print(f"Random vs Distance concepts: {random_vs_distance.cka:.4f}")
    print(f"Cross-domain effect: {cross_snr_dist:+.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print("\nIf FRBs encode physical semantics:")
    print("  - Physics-matched ordering >> random ordering")
    print("  - Same-domain matching >> cross-domain matching")
    print("  - FRB effect >> noise effect")

    # Compute key metrics
    frb_physics_effect = (dm_improvement + snr_improvement) / 2
    noise_physics_effect = noise_dm_diff
    domain_specificity = (dm_improvement - cross_dm_int + snr_improvement - cross_snr_dist) / 2

    print(f"\nKey Metrics:")
    print(f"  FRB physics-matching effect: {frb_physics_effect:+.4f}")
    print(f"  Noise 'physics-matching' effect: {noise_physics_effect:+.4f}")
    print(f"  Domain specificity: {domain_specificity:+.4f}")

    if frb_physics_effect > noise_physics_effect + 0.02:
        print("\n** FRBs show physics-dependent alignment that noise lacks **")
    elif abs(frb_physics_effect - noise_physics_effect) < 0.02:
        print("\n** FRBs and noise show similar (non-)effects - no physics signal **")
    else:
        print("\n** Inconclusive - effects are small or reversed **")

    results = {
        "experiment": "exp8_physical_semantics",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "dm_alignment": {
            "physics_matched_cka": float(dm_matched_cka.cka),
            "random_order_cka": float(dm_random_cka.cka),
            "improvement": float(dm_improvement),
        },
        "snr_alignment": {
            "physics_matched_cka": float(snr_matched_cka.cka),
            "random_order_cka": float(snr_random_cka.cka),
            "improvement": float(snr_improvement),
        },
        "noise_control": {
            "dm_matched_cka": float(noise_dm_matched_cka.cka),
            "random_cka": float(noise_dm_random_cka.cka),
            "difference": float(noise_dm_diff),
        },
        "cross_validation": {
            "dm_sorted_vs_intensity": float(dm_sorted_vs_intensity.cka),
            "snr_sorted_vs_distance": float(snr_sorted_vs_distance.cka),
            "cross_dm_intensity_effect": float(cross_dm_int),
            "cross_snr_distance_effect": float(cross_snr_dist),
        },
        "summary": {
            "frb_physics_effect": float(frb_physics_effect),
            "noise_physics_effect": float(noise_physics_effect),
            "domain_specificity": float(domain_specificity),
        },
    }

    output_path = results_dir / "exp8_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
