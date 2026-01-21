#!/usr/bin/env python3
"""Experiment 3: Deep Investigation of Whisper-FRB Alignment.

The raw CKA between FRB features and Whisper embeddings is 0.9884 -
dramatically higher than random baseline (0.18) or CLIP (0.18).

This experiment investigates:
1. Is this a fluke of the specific concepts used?
2. Which FRB features drive the correlation?
3. Does the alignment hold with different Whisper models?
4. Is there structure in which FRBs align best?

Usage:
    poetry run python experiments/astronomy/exp3_whisper_investigation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import compute_cka

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features, get_feature_dimension


def extract_whisper_with_concepts(
    concepts: list[str],
    backend,
    model_name: str = "openai/whisper-base",
):
    """Extract Whisper embeddings for specific concepts."""
    from transformers import WhisperModel, WhisperProcessor
    import torch

    model = WhisperModel.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
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
    embeddings_np = embeddings.detach().cpu().numpy()

    return backend.array(embeddings_np)


def run_investigation():
    """Deep investigation of Whisper-FRB alignment."""
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 3: Whisper-FRB Deep Investigation")
    print("=" * 60)
    print()

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    print(f"Found {len(frb_files)} FRB files")

    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"Loaded {n_frbs} FRBs")

    # Extract features
    frb_features = batch_extract_features(waterfalls, backend)
    feature_names = [
        "band_0_mean", "band_0_std", "band_1_mean", "band_1_std",
        "band_2_mean", "band_2_std", "band_3_mean", "band_3_std",
        "band_4_mean", "band_4_std", "band_5_mean", "band_5_std",
        "band_6_mean", "band_6_std", "band_7_mean", "band_7_std",
        "ts_mean", "ts_std", "ts_max", "ts_peak_location",
        "spec_entropy", "spec_peak_freq", "spec_bandwidth",
        "aspect_ratio", "total_intensity", "sparsity",
    ]

    results = {
        "experiment": "exp3_whisper_investigation",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }

    # --- Test 1: Different concept sets ---
    print()
    print("=" * 40)
    print("TEST 1: Different Concept Sets")
    print("=" * 40)

    concept_sets = {
        "astronomy": [
            "cosmic burst", "radio signal", "electromagnetic wave",
            "frequency spectrum", "dispersion measure", "neutron star",
            *[f"signal {i}" for i in range(n_frbs - 6)]
        ][:n_frbs],
        "random_words": [
            "apple", "banana", "car", "dog", "elephant", "fish",
            *[f"word{i}" for i in range(n_frbs - 6)]
        ][:n_frbs],
        "numbers": [str(i) for i in range(n_frbs)],
        "single_letter": ["a"] * n_frbs,
        "music": [
            "bass note", "drum beat", "melody", "harmony", "rhythm",
            *[f"tone {i}" for i in range(n_frbs - 5)]
        ][:n_frbs],
    }

    for name, concepts in concept_sets.items():
        print(f"\n--- Concept set: {name} ---")
        whisper_emb = extract_whisper_with_concepts(concepts, backend)
        cka_result = compute_cka(frb_features, whisper_emb, backend=backend)
        print(f"  Raw CKA: {cka_result.cka:.4f}")

        results["tests"].append({
            "test": "concept_set",
            "name": name,
            "raw_cka": float(cka_result.cka),
        })

    # --- Test 2: Feature ablation ---
    print()
    print("=" * 40)
    print("TEST 2: Feature Ablation")
    print("=" * 40)

    # Get baseline Whisper embeddings
    baseline_concepts = [f"signal {i}" for i in range(n_frbs)]
    whisper_baseline = extract_whisper_with_concepts(baseline_concepts, backend)

    # Test each feature group
    feature_groups = {
        "frequency_bands": list(range(16)),  # First 16 features
        "time_series": [16, 17, 18, 19],  # ts_mean, ts_std, ts_max, ts_peak_location
        "spectral": [20, 21, 22],  # spec_entropy, spec_peak_freq, spec_bandwidth
        "morphological": [23, 24, 25],  # aspect_ratio, total_intensity, sparsity
    }

    frb_np = backend.tolist(frb_features)
    frb_np = np.array(frb_np)

    for group_name, indices in feature_groups.items():
        # Test with ONLY this group
        group_features = backend.array(frb_np[:, indices])
        cka_result = compute_cka(group_features, whisper_baseline, backend=backend)
        print(f"\n--- Only {group_name} (dims {indices[0]}-{indices[-1]}) ---")
        print(f"  Raw CKA: {cka_result.cka:.4f}")

        results["tests"].append({
            "test": "feature_ablation",
            "group": group_name,
            "indices": indices,
            "raw_cka": float(cka_result.cka),
        })

    # --- Test 3: Per-FRB alignment ---
    print()
    print("=" * 40)
    print("TEST 3: Per-FRB Gram Matrix Analysis")
    print("=" * 40)

    # Compute Gram matrices
    frb_gram = backend.matmul(frb_features, backend.transpose(frb_features))
    whisper_gram = backend.matmul(whisper_baseline, backend.transpose(whisper_baseline))

    # Center the Gram matrices (as CKA does)
    n = frb_gram.shape[0]
    centering = backend.eye(n) - backend.ones((n, n)) / n
    frb_gram_centered = backend.matmul(backend.matmul(centering, frb_gram), centering)
    whisper_gram_centered = backend.matmul(backend.matmul(centering, whisper_gram), centering)

    # Look at diagonal values (self-similarity)
    frb_diag = [float(backend.tolist(frb_gram_centered)[i][i]) for i in range(n)]
    whisper_diag = [float(backend.tolist(whisper_gram_centered)[i][i]) for i in range(n)]

    print("\nGram matrix diagonal correlation:")
    frb_diag_np = np.array(frb_diag)
    whisper_diag_np = np.array(whisper_diag)
    diag_corr = np.corrcoef(frb_diag_np, whisper_diag_np)[0, 1]
    print(f"  Diagonal correlation: {diag_corr:.4f}")

    # Full matrix correlation
    frb_flat = np.array(backend.tolist(frb_gram_centered)).flatten()
    whisper_flat = np.array(backend.tolist(whisper_gram_centered)).flatten()
    full_corr = np.corrcoef(frb_flat, whisper_flat)[0, 1]
    print(f"  Full Gram correlation: {full_corr:.4f}")

    results["tests"].append({
        "test": "gram_analysis",
        "diagonal_correlation": float(diag_corr),
        "full_gram_correlation": float(full_corr),
    })

    # --- Test 4: Whisper model variants ---
    print()
    print("=" * 40)
    print("TEST 4: Different Whisper Models")
    print("=" * 40)

    whisper_models = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
    ]

    for model_name in whisper_models:
        print(f"\n--- {model_name} ---")
        try:
            whisper_emb = extract_whisper_with_concepts(baseline_concepts, backend, model_name)
            cka_result = compute_cka(frb_features, whisper_emb, backend=backend)
            print(f"  Raw CKA: {cka_result.cka:.4f}")
            print(f"  Embedding dim: {whisper_emb.shape[1]}")

            results["tests"].append({
                "test": "whisper_model",
                "model": model_name,
                "raw_cka": float(cka_result.cka),
                "embed_dim": int(whisper_emb.shape[1]),
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("INVESTIGATION SUMMARY")
    print("=" * 60)

    # Find the key findings
    concept_results = [t for t in results["tests"] if t["test"] == "concept_set"]
    print("\nConcept set CKA values:")
    for r in concept_results:
        print(f"  {r['name']}: {r['raw_cka']:.4f}")

    ablation_results = [t for t in results["tests"] if t["test"] == "feature_ablation"]
    print("\nFeature group CKA values:")
    for r in ablation_results:
        print(f"  {r['group']}: {r['raw_cka']:.4f}")

    # Save results
    output_path = results_dir / "exp3_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_investigation()
