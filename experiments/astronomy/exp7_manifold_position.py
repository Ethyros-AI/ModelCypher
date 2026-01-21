#!/usr/bin/env python3
"""Experiment 7: Manifold Position Analysis.

Inspired by Arrival: If knowledge has universal geometric structure,
white noise is a concept too ("randomness"). The question isn't
"does FRB correlate with X?" but "WHERE on the manifold do FRBs live?"

This experiment:
1. Maps FRB features to semantic concept neighborhoods
2. Finds which concepts FRBs are geometrically adjacent to
3. Compares FRB position vs noise position on the manifold
4. Tests trajectory coherence (do FRBs form a path or random scatter?)

Usage:
    poetry run python experiments/astronomy/exp7_manifold_position.py
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


def compute_concept_distances(features, concept_embeddings, backend):
    """Compute distance from features to each concept embedding.

    Uses CKA as similarity metric (higher = closer).
    Returns per-sample similarity to the concept space.
    """
    # Compute Gram matrix of features
    K_features = backend.matmul(features, backend.transpose(features, (1, 0)))

    # Compute Gram matrix of concepts
    K_concepts = backend.matmul(concept_embeddings, backend.transpose(concept_embeddings, (1, 0)))

    # Center the Gram matrices
    n = K_features.shape[0]
    H = backend.eye(n) - backend.ones((n, n)) / n
    K_features_c = backend.matmul(backend.matmul(H, K_features), H)
    K_concepts_c = backend.matmul(backend.matmul(H, K_concepts), H)

    # Return the centered Gram matrices for analysis
    return K_features_c, K_concepts_c


def generate_noise_features(n_samples: int, backend, seed: int = 42):
    """Generate features from white noise (for comparison)."""
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


def compute_trajectory_coherence(features, backend):
    """Measure if sequential samples form coherent trajectory.

    Compares:
    1. Sequential distances (sample i to i+1)
    2. Random pair distances

    If trajectory is coherent, sequential distances should be
    smaller than random distances (smooth path vs random scatter).
    """
    n = features.shape[0]

    # Compute pairwise distances
    # Using Euclidean distance in feature space
    distances = np.zeros((n, n))
    features_np = np.array(backend.tolist(features))

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(features_np[i] - features_np[j])

    # Sequential distances (i to i+1)
    sequential_dists = [distances[i, i+1] for i in range(n-1)]

    # Random pair distances (excluding sequential)
    random_dists = []
    for i in range(n):
        for j in range(i+2, n):  # Skip adjacent pairs
            random_dists.append(distances[i, j])

    return {
        "sequential_mean": float(np.mean(sequential_dists)),
        "sequential_std": float(np.std(sequential_dists)),
        "random_mean": float(np.mean(random_dists)),
        "random_std": float(np.std(random_dists)),
        "coherence_ratio": float(np.mean(sequential_dists) / np.mean(random_dists)),
    }


def find_concept_neighborhood(features, concept_library, backend):
    """Find which semantic concepts are closest to the feature space.

    concept_library: dict mapping concept names to concept strings

    Returns ranked list of concepts by CKA similarity.
    """
    n_samples = features.shape[0]
    results = []

    for category, concepts in concept_library.items():
        # Pad or truncate to match n_samples
        if len(concepts) < n_samples:
            concepts = concepts * (n_samples // len(concepts) + 1)
        concepts = concepts[:n_samples]

        embeddings = extract_concept_embeddings(concepts, backend)
        cka = compute_cka(features, embeddings, backend=backend)

        results.append({
            "category": category,
            "cka": float(cka.cka),
            "concepts_sample": concepts[:3],
        })

    # Sort by CKA (highest first = closest on manifold)
    results.sort(key=lambda x: x["cka"], reverse=True)
    return results


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 7: Manifold Position Analysis")
    print("=" * 60)
    print("\nQuestion: WHERE on the universal knowledge manifold do FRBs live?")
    print("(Not: do FRBs correlate with X?)")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    # Generate noise for comparison
    noise_np = generate_noise_features(n_frbs, backend)

    # Define concept library - semantic categories to test proximity
    # These represent different REGIONS of the knowledge manifold
    concept_library = {
        # Physical/Natural phenomena
        "physical_quantities": [
            f"distance of {i * 100} meters" for i in range(n_frbs)
        ],
        "energy_levels": [
            f"energy level {i}" for i in range(n_frbs)
        ],
        "frequencies": [
            f"frequency of {100 + i * 50} hertz" for i in range(n_frbs)
        ],
        "time_durations": [
            f"duration of {i * 10} milliseconds" for i in range(n_frbs)
        ],

        # Sequential/Ordering concepts
        "sequential_numbers": [
            f"number {i}" for i in range(n_frbs)
        ],
        "ordinal_positions": [
            f"the {i+1}th item" for i in range(n_frbs)
        ],
        "temporal_sequence": [
            f"event {i} in sequence" for i in range(n_frbs)
        ],

        # Abstract concepts
        "randomness": [
            f"random noise sample {i}" for i in range(n_frbs)
        ],
        "entropy": [
            f"entropy level {i}" for i in range(n_frbs)
        ],
        "information": [
            f"information content {i} bits" for i in range(n_frbs)
        ],

        # Signal/Communication
        "radio_signals": [
            f"radio signal {i}" for i in range(n_frbs)
        ],
        "pulses": [
            f"pulse number {i}" for i in range(n_frbs)
        ],
        "transmissions": [
            f"transmission {i}" for i in range(n_frbs)
        ],

        # Natural/Cosmic
        "cosmic_events": [
            f"cosmic event {i}" for i in range(n_frbs)
        ],
        "stellar_phenomena": [
            f"star burst {i}" for i in range(n_frbs)
        ],
        "distances_cosmic": [
            f"distance of {i * 100} megaparsecs" for i in range(n_frbs)
        ],
    }

    print("\n" + "=" * 40)
    print("PART 1: CONCEPT NEIGHBORHOOD ANALYSIS")
    print("=" * 40)
    print("\nFinding which semantic concepts FRBs are closest to...")

    frb_neighborhood = find_concept_neighborhood(
        backend.array(frb_np), concept_library, backend
    )

    print("\nFRB Feature Neighborhood (sorted by proximity):")
    for item in frb_neighborhood:
        print(f"  {item['category']}: CKA = {item['cka']:.4f}")

    print("\n" + "-" * 40)
    print("Finding which semantic concepts NOISE is closest to...")

    noise_neighborhood = find_concept_neighborhood(
        backend.array(noise_np), concept_library, backend
    )

    print("\nNoise Feature Neighborhood (sorted by proximity):")
    for item in noise_neighborhood:
        print(f"  {item['category']}: CKA = {item['cka']:.4f}")

    print("\n" + "=" * 40)
    print("PART 2: DIFFERENTIAL MANIFOLD POSITION")
    print("=" * 40)
    print("\nComparing FRB vs Noise position for each concept category:")

    differential = []
    for frb_item in frb_neighborhood:
        noise_item = next(
            n for n in noise_neighborhood if n["category"] == frb_item["category"]
        )
        diff = frb_item["cka"] - noise_item["cka"]
        differential.append({
            "category": frb_item["category"],
            "frb_cka": frb_item["cka"],
            "noise_cka": noise_item["cka"],
            "difference": diff,
        })

    # Sort by difference (where FRBs are most distinct from noise)
    differential.sort(key=lambda x: x["difference"], reverse=True)

    print("\nCategories where FRBs differ most from noise:")
    for item in differential:
        sign = "+" if item["difference"] > 0 else ""
        print(f"  {item['category']}: FRB={item['frb_cka']:.4f}, "
              f"Noise={item['noise_cka']:.4f}, Δ={sign}{item['difference']:.4f}")

    print("\n" + "=" * 40)
    print("PART 3: TRAJECTORY COHERENCE")
    print("=" * 40)
    print("\nDo FRBs form a coherent path or random scatter?")
    print("(Sequential samples should be closer than random pairs if coherent)")

    frb_coherence = compute_trajectory_coherence(backend.array(frb_np), backend)
    noise_coherence = compute_trajectory_coherence(backend.array(noise_np), backend)

    print(f"\nFRB Trajectory:")
    print(f"  Sequential distance (mean): {frb_coherence['sequential_mean']:.4f}")
    print(f"  Random pair distance (mean): {frb_coherence['random_mean']:.4f}")
    print(f"  Coherence ratio: {frb_coherence['coherence_ratio']:.4f}")
    print(f"  (Ratio < 1.0 means trajectory is coherent)")

    print(f"\nNoise Trajectory:")
    print(f"  Sequential distance (mean): {noise_coherence['sequential_mean']:.4f}")
    print(f"  Random pair distance (mean): {noise_coherence['random_mean']:.4f}")
    print(f"  Coherence ratio: {noise_coherence['coherence_ratio']:.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Find where FRBs are most distinct
    top_frb_categories = [d["category"] for d in differential[:3] if d["difference"] > 0.05]
    top_noise_categories = [d["category"] for d in differential[-3:] if d["difference"] < -0.05]

    print("\nFRBs are geometrically closer to:")
    if top_frb_categories:
        for cat in top_frb_categories:
            print(f"  - {cat}")
    else:
        print("  (No strong differential signal)")

    print("\nNoise is geometrically closer to:")
    if top_noise_categories:
        for cat in top_noise_categories:
            print(f"  - {cat}")
    else:
        print("  (No strong differential signal)")

    if frb_coherence["coherence_ratio"] < noise_coherence["coherence_ratio"]:
        print("\nFRBs show MORE trajectory coherence than noise.")
        print("(Chronological FRBs form a smoother path through feature space)")
    else:
        print("\nFRBs show similar or less trajectory coherence than noise.")

    results = {
        "experiment": "exp7_manifold_position",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_frbs,
        "frb_neighborhood": frb_neighborhood,
        "noise_neighborhood": noise_neighborhood,
        "differential_analysis": differential,
        "trajectory_coherence": {
            "frb": frb_coherence,
            "noise": noise_coherence,
        },
    }

    output_path = results_dir / "exp7_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
