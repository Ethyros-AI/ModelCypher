#!/usr/bin/env python3
"""Experiment 6: Ordering Investigation.

With "unique_numbers" concepts, FRBs show CKA = 0.948 vs noise = 0.600.
This 0.35 difference suggests FRBs may have sequential structure.

The FRB files are sorted chronologically by detection date.
This experiment tests:
1. Is there temporal structure in FRB features?
2. Do physical properties (DM, SNR) correlate with detection order?
3. Does shuffling FRBs eliminate the CKA advantage?

Usage:
    poetry run python experiments/astronomy/exp6_ordering_investigation.py
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


def extract_whisper_embeddings(concepts: list[str], backend):
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
    print("Experiment 6: Ordering Investigation")
    print("=" * 60)

    # Load FRBs (sorted by filename = chronological order)
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"Loaded {n_frbs} FRBs in chronological order")

    # Get features
    frb_features_arr = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features_arr))

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])
    indices = np.arange(n_frbs)

    print("\n" + "=" * 40)
    print("TEMPORAL CORRELATIONS")
    print("=" * 40)

    # Check if physical properties correlate with order
    dm_order_corr = np.corrcoef(indices, dms)[0, 1]
    snr_order_corr = np.corrcoef(indices, snrs)[0, 1]
    print(f"\nDetection order vs DM:  r = {dm_order_corr:.4f}")
    print(f"Detection order vs SNR: r = {snr_order_corr:.4f}")

    # Check feature correlations with order
    print("\nFeature correlations with detection order:")
    feature_names = [
        "band_0_mean", "band_0_std", "ts_mean", "ts_std", "ts_max",
        "ts_peak_location", "spec_entropy", "total_intensity"
    ]
    feature_indices = [0, 1, 16, 17, 18, 19, 20, 24]

    order_correlations = {}
    for name, idx in zip(feature_names, feature_indices):
        corr = np.corrcoef(indices, frb_np[:, idx])[0, 1]
        if not np.isnan(corr):
            order_correlations[name] = float(corr)
            print(f"  {name}: r = {corr:.4f}")

    print("\n" + "=" * 40)
    print("SHUFFLING TEST")
    print("=" * 40)

    # Get whisper embeddings
    concepts = [f"number {i * 7 + 3}" for i in range(n_frbs)]
    whisper_emb = extract_whisper_embeddings(concepts, backend)

    # Original CKA
    original_cka = compute_cka(backend.array(frb_np), whisper_emb, backend=backend)
    print(f"\nOriginal FRB CKA (chronological): {original_cka.cka:.4f}")

    # Shuffled CKAs
    rng = np.random.default_rng(42)
    shuffled_ckas = []

    print("\nShuffled FRB CKAs (10 random permutations):")
    for i in range(10):
        perm = rng.permutation(n_frbs)
        shuffled_features = backend.array(frb_np[perm])
        shuffled_cka = compute_cka(shuffled_features, whisper_emb, backend=backend)
        shuffled_ckas.append(float(shuffled_cka.cka))
        print(f"  Shuffle {i+1}: {shuffled_cka.cka:.4f}")

    print(f"\nShuffled mean: {np.mean(shuffled_ckas):.4f}")
    print(f"Shuffled std:  {np.std(shuffled_ckas):.4f}")

    # Noise comparison
    noise_np = generate_noise_features(n_frbs, backend)
    noise_cka = compute_cka(backend.array(noise_np), whisper_emb, backend=backend)
    print(f"\nNoise CKA: {noise_cka.cka:.4f}")

    print("\n" + "=" * 40)
    print("DM-ORDERED TEST")
    print("=" * 40)

    # Sort FRBs by DM instead of chronologically
    dm_order = np.argsort(dms)
    dm_sorted_features = backend.array(frb_np[dm_order])
    dm_sorted_cka = compute_cka(dm_sorted_features, whisper_emb, backend=backend)
    print(f"\nFRBs sorted by DM: CKA = {dm_sorted_cka.cka:.4f}")

    # SNR sorted
    snr_order = np.argsort(snrs)
    snr_sorted_features = backend.array(frb_np[snr_order])
    snr_sorted_cka = compute_cka(snr_sorted_features, whisper_emb, backend=backend)
    print(f"FRBs sorted by SNR: CKA = {snr_sorted_cka.cka:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nOriginal (chronological): {original_cka.cka:.4f}")
    print(f"Shuffled (mean):          {np.mean(shuffled_ckas):.4f}")
    print(f"DM-sorted:                {dm_sorted_cka.cka:.4f}")
    print(f"SNR-sorted:               {snr_sorted_cka.cka:.4f}")
    print(f"Noise:                    {noise_cka.cka:.4f}")

    if original_cka.cka > np.mean(shuffled_ckas) + 2 * np.std(shuffled_ckas):
        print("\n** FINDING: Chronological order matters! **")
    else:
        print("\n** FINDING: Order doesn't significantly affect CKA **")

    results = {
        "experiment": "exp6_ordering_investigation",
        "timestamp": datetime.now().isoformat(),
        "temporal_correlations": {
            "dm_vs_order": float(dm_order_corr),
            "snr_vs_order": float(snr_order_corr),
            "features_vs_order": order_correlations,
        },
        "cka_results": {
            "original_chronological": float(original_cka.cka),
            "shuffled_mean": float(np.mean(shuffled_ckas)),
            "shuffled_std": float(np.std(shuffled_ckas)),
            "dm_sorted": float(dm_sorted_cka.cka),
            "snr_sorted": float(snr_sorted_cka.cka),
            "noise": float(noise_cka.cka),
        },
    }

    output_path = results_dir / "exp6_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
