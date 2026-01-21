#!/usr/bin/env python3
"""Experiment 5: Concept Dependency Analysis.

The original exp2 showed CKA = 0.9884 with Whisper.
But exp3 showed it varies dramatically with concept choice.
And exp4 showed noise gives HIGHER CKA than FRBs.

This experiment:
1. Uses the EXACT concepts from the original exp2
2. Tests noise features with those same concepts
3. Determines if the 0.99 result is reproducible and noise-specific

Usage:
    poetry run python experiments/astronomy/exp5_concept_dependency.py
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
from shared.reference_embeddings import DEFAULT_CONCEPTS


def generate_noise_features(n_samples: int, backend, seed: int = 42):
    """Generate features from pure white noise."""
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

    return backend.array(features)


def extract_whisper_embeddings(concepts: list[str], backend):
    """Extract Whisper embeddings exactly as in reference_embeddings.py."""
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


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 5: Concept Dependency Analysis")
    print("=" * 60)

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"Loaded {n_frbs} FRBs")

    # Get FRB features
    frb_features = batch_extract_features(waterfalls, backend)

    # Get noise features
    print("Generating noise features...")
    noise_features = generate_noise_features(n_frbs, backend)

    # The EXACT concepts from reference_embeddings.py
    original_concepts = list(DEFAULT_CONCEPTS[:n_frbs])
    print(f"\nUsing {len(original_concepts)} original concepts:")
    print(f"  First 5: {original_concepts[:5]}")

    # Get Whisper embeddings with original concepts
    print("\nExtracting Whisper embeddings with original concepts...")
    whisper_original = extract_whisper_embeddings(original_concepts, backend)

    # Test 1: FRB vs Whisper (original concepts)
    frb_cka = compute_cka(frb_features, whisper_original, backend=backend)
    print(f"\n=== RESULTS ===")
    print(f"FRB features vs Whisper (original concepts): CKA = {frb_cka.cka:.4f}")

    # Test 2: Noise vs Whisper (original concepts)
    noise_cka = compute_cka(noise_features, whisper_original, backend=backend)
    print(f"Noise features vs Whisper (original concepts): CKA = {noise_cka.cka:.4f}")

    # Test 3: Different concept variations
    print("\n=== CONCEPT VARIATIONS ===")

    concept_variations = {
        "original": original_concepts,
        "padded_original": original_concepts + [f"another {original_concepts[i % len(DEFAULT_CONCEPTS)]}" for i in range(max(0, n_frbs - len(DEFAULT_CONCEPTS)))],
        "sequential_numbers": [str(i) for i in range(n_frbs)],
        "repeated_phrase": ["radio burst from space"] * n_frbs,
        "unique_numbers": [f"number {i * 7 + 3}" for i in range(n_frbs)],
    }

    results = {
        "experiment": "exp5_concept_dependency",
        "timestamp": datetime.now().isoformat(),
        "n_frbs": n_frbs,
        "tests": [],
    }

    for name, concepts in concept_variations.items():
        print(f"\n--- {name} ---")
        print(f"  First 3: {concepts[:3]}")

        whisper_emb = extract_whisper_embeddings(concepts, backend)

        frb_cka_result = compute_cka(frb_features, whisper_emb, backend=backend)
        noise_cka_result = compute_cka(noise_features, whisper_emb, backend=backend)

        print(f"  FRB CKA:   {frb_cka_result.cka:.4f}")
        print(f"  Noise CKA: {noise_cka_result.cka:.4f}")
        print(f"  Difference (FRB - Noise): {frb_cka_result.cka - noise_cka_result.cka:.4f}")

        results["tests"].append({
            "concepts": name,
            "frb_cka": float(frb_cka_result.cka),
            "noise_cka": float(noise_cka_result.cka),
            "difference": float(frb_cka_result.cka - noise_cka_result.cka),
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test in results["tests"]:
        indicator = "FRB > Noise" if test["difference"] > 0.01 else "FRB ≈ Noise" if abs(test["difference"]) < 0.01 else "Noise > FRB"
        print(f"{test['concepts']:<20}: FRB={test['frb_cka']:.3f}, Noise={test['noise_cka']:.3f} ({indicator})")

    output_path = results_dir / "exp5_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
