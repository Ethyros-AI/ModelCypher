#!/usr/bin/env python3
"""Experiment 4: Time Series Feature Deep Dive.

The time_series features (ts_mean, ts_std, ts_max, ts_peak_location) show
CKA = 0.74 with Whisper - the highest of any feature group.

This experiment investigates:
1. What do these features actually capture?
2. Are they correlated with FRB physical properties (DM, SNR)?
3. What is the distribution of these features across FRBs?
4. Can we find similar patterns in pure white noise?

Usage:
    poetry run python experiments/astronomy/exp4_time_series_analysis.py
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
from shared.feature_extraction import batch_extract_features


def generate_noise_features(n_samples: int, backend, seed: int = 42):
    """Generate features from pure white noise 'spectrograms'.

    Creates synthetic spectrograms with no structure, extracts the same
    features as we do for FRBs, to see if CKA with Whisper is still high.
    """
    from shared.feature_extraction import extract_frb_features

    rng = np.random.default_rng(seed)

    features = []
    for i in range(n_samples):
        # Create random spectrogram
        n_freq, n_time = 256, 1024
        waterfall = rng.standard_normal((n_freq, n_time)).astype(np.float32)
        waterfall = backend.array(waterfall)

        # Create time series and spectrum
        time_series = backend.array(rng.standard_normal(n_time).astype(np.float32))
        spectrum = backend.array(rng.standard_normal(n_freq).astype(np.float32))

        frb_feat = extract_frb_features(
            waterfall, time_series, spectrum, backend,
            tns_name=f"noise_{i}"
        )
        features.append(backend.tolist(frb_feat.features))

    return backend.array(features)


def run_analysis():
    """Deep analysis of time series features."""
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 4: Time Series Feature Deep Dive")
    print("=" * 60)
    print()

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"Loaded {n_frbs} FRBs")

    # Extract features
    frb_features = batch_extract_features(waterfalls, backend)
    frb_np = np.array(backend.tolist(frb_features))

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
        "experiment": "exp4_time_series_analysis",
        "timestamp": datetime.now().isoformat(),
    }

    # --- Feature Statistics ---
    print()
    print("=" * 40)
    print("TIME SERIES FEATURE STATISTICS")
    print("=" * 40)

    ts_indices = [16, 17, 18, 19]
    ts_names = ["ts_mean", "ts_std", "ts_max", "ts_peak_location"]

    ts_stats = {}
    for idx, name in zip(ts_indices, ts_names):
        values = frb_np[:, idx]
        ts_stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values.tolist(),
        }
        print(f"\n{name}:")
        print(f"  Mean: {ts_stats[name]['mean']:.4f}")
        print(f"  Std:  {ts_stats[name]['std']:.4f}")
        print(f"  Range: [{ts_stats[name]['min']:.4f}, {ts_stats[name]['max']:.4f}]")

    results["ts_statistics"] = ts_stats

    # --- Correlation with physical properties ---
    print()
    print("=" * 40)
    print("CORRELATION WITH PHYSICAL PROPERTIES")
    print("=" * 40)

    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])

    correlations = {}
    for idx, name in zip(ts_indices, ts_names):
        values = frb_np[:, idx]
        dm_corr = np.corrcoef(values, dms)[0, 1]
        snr_corr = np.corrcoef(values, snrs)[0, 1]
        correlations[name] = {
            "dm_correlation": float(dm_corr),
            "snr_correlation": float(snr_corr),
        }
        print(f"\n{name}:")
        print(f"  DM correlation:  {dm_corr:.4f}")
        print(f"  SNR correlation: {snr_corr:.4f}")

    results["physical_correlations"] = correlations

    # --- Compare to white noise ---
    print()
    print("=" * 40)
    print("COMPARISON TO WHITE NOISE")
    print("=" * 40)

    print("\nGenerating white noise features...")
    noise_features = generate_noise_features(n_frbs, backend)
    noise_np = np.array(backend.tolist(noise_features))

    print("\nWhite noise time series statistics:")
    noise_ts_stats = {}
    for idx, name in zip(ts_indices, ts_names):
        values = noise_np[:, idx]
        noise_ts_stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        print(f"  {name}: mean={noise_ts_stats[name]['mean']:.4f}, std={noise_ts_stats[name]['std']:.4f}")

    results["noise_ts_statistics"] = noise_ts_stats

    # CKA of noise features with Whisper
    print("\nComputing CKA of noise features with Whisper...")
    from transformers import WhisperModel, WhisperProcessor
    import torch

    model = WhisperModel.from_pretrained("openai/whisper-base")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    tokenizer = processor.tokenizer

    concepts = [f"signal {i}" for i in range(n_frbs)]
    all_embeds = []
    for concept in concepts:
        tokens = tokenizer(concept, return_tensors="pt").input_ids
        with torch.no_grad():
            embed_layer = model.decoder.embed_tokens
            embeds = embed_layer(tokens)
            pooled = embeds.mean(dim=1)
            all_embeds.append(pooled)
    whisper_emb = torch.cat(all_embeds, dim=0)
    whisper_emb = backend.array(whisper_emb.detach().cpu().numpy())

    # CKA: FRB vs Whisper
    frb_cka = compute_cka(frb_features, whisper_emb, backend=backend)
    print(f"\nFRB features vs Whisper: CKA = {frb_cka.cka:.4f}")

    # CKA: Noise vs Whisper
    noise_cka = compute_cka(noise_features, whisper_emb, backend=backend)
    print(f"Noise features vs Whisper: CKA = {noise_cka.cka:.4f}")

    # CKA: FRB time series only vs Whisper
    frb_ts = backend.array(frb_np[:, ts_indices])
    frb_ts_cka = compute_cka(frb_ts, whisper_emb, backend=backend)
    print(f"FRB time series only vs Whisper: CKA = {frb_ts_cka.cka:.4f}")

    # CKA: Noise time series only vs Whisper
    noise_ts = backend.array(noise_np[:, ts_indices])
    noise_ts_cka = compute_cka(noise_ts, whisper_emb, backend=backend)
    print(f"Noise time series only vs Whisper: CKA = {noise_ts_cka.cka:.4f}")

    results["cka_comparison"] = {
        "frb_all_vs_whisper": float(frb_cka.cka),
        "noise_all_vs_whisper": float(noise_cka.cka),
        "frb_ts_vs_whisper": float(frb_ts_cka.cka),
        "noise_ts_vs_whisper": float(noise_ts_cka.cka),
    }

    # --- Rank/variance analysis ---
    print()
    print("=" * 40)
    print("VARIANCE STRUCTURE ANALYSIS")
    print("=" * 40)

    # Singular value decomposition of each feature matrix
    frb_centered = frb_np - frb_np.mean(axis=0)
    noise_centered = noise_np - noise_np.mean(axis=0)
    whisper_np = np.array(backend.tolist(whisper_emb))
    whisper_centered = whisper_np - whisper_np.mean(axis=0)

    frb_svd = np.linalg.svd(frb_centered, compute_uv=False)
    noise_svd = np.linalg.svd(noise_centered, compute_uv=False)
    whisper_svd = np.linalg.svd(whisper_centered, compute_uv=False)

    # Normalized singular values (explains variance fraction)
    frb_sv_norm = frb_svd / frb_svd.sum()
    noise_sv_norm = noise_svd / noise_svd.sum()
    whisper_sv_norm = whisper_svd / whisper_svd.sum()

    print("\nTop 5 singular values (normalized):")
    print(f"  FRB:     {frb_sv_norm[:5]}")
    print(f"  Noise:   {noise_sv_norm[:5]}")
    print(f"  Whisper: {whisper_sv_norm[:5]}")

    # Effective rank
    def effective_rank(sv_norm):
        entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
        return np.exp(entropy)

    frb_rank = effective_rank(frb_sv_norm)
    noise_rank = effective_rank(noise_sv_norm)
    whisper_rank = effective_rank(whisper_sv_norm)

    print(f"\nEffective rank:")
    print(f"  FRB:     {frb_rank:.2f}")
    print(f"  Noise:   {noise_rank:.2f}")
    print(f"  Whisper: {whisper_rank:.2f}")

    results["variance_structure"] = {
        "frb_top5_sv": frb_sv_norm[:5].tolist(),
        "noise_top5_sv": noise_sv_norm[:5].tolist(),
        "whisper_top5_sv": whisper_sv_norm[:5].tolist(),
        "frb_effective_rank": float(frb_rank),
        "noise_effective_rank": float(noise_rank),
        "whisper_effective_rank": float(whisper_rank),
    }

    # --- Summary ---
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print()
    print("If noise features also show high CKA with Whisper,")
    print("the FRB result is likely a variance structure artifact.")
    print("If FRB shows significantly higher CKA than noise,")
    print("there may be genuine structure in FRB time series.")

    # Save results
    output_path = results_dir / "exp4_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_analysis()
