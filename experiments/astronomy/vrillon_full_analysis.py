#!/usr/bin/env python3
"""Vrillon Full-Resolution Spectrogram Analysis

Analyze the Vrillon spectrogram at full resolution for metrics that scale well,
and at multiple resolutions to check for scale-invariant encodings.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from modelcypher.backends import initialize_default_backend
initialize_default_backend()

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank

# Constants
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)

CONSTANTS = {
    "π": PI, "e": E, "φ": PHI, "√2": SQRT2,
    "π/2": PI/2, "2π": 2*PI, "π²": PI**2,
    "e²": E**2, "φ²": PHI**2,
    "π/e": PI/E, "e/π": E/PI, "φ/π": PHI/PI,
    "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0,
    "7": 7.0, "10": 10.0,
}


def percent_error(measured: float, expected: float) -> float:
    if expected == 0:
        return float('inf')
    return abs(measured - expected) / expected * 100


def find_closest(value: float) -> tuple[str, float, float]:
    best_name, best_val, best_err = "none", 0.0, float('inf')
    for name, const in CONSTANTS.items():
        err = percent_error(value, const)
        if err < best_err:
            best_name, best_val, best_err = name, const, err
    return best_name, best_val, best_err


def analyze_spectrogram(matrix: np.ndarray, name: str) -> dict:
    """Compute all metrics for a spectrogram matrix."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"Shape: {matrix.shape}")
    print(f"{'='*60}")

    backend = get_default_backend()
    er = EffectiveRank(backend=backend)
    arr = backend.array(matrix.astype(np.float32))
    er_result = er.compute(arr)

    # SVD analysis
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_sq = S ** 2
    spectral_pr = (np.sum(S_sq) ** 2) / np.sum(S_sq ** 2)

    # SV ratios
    sv_ratios = []
    for i in range(min(10, len(S) - 1)):
        if S[i+1] > 1e-10:
            sv_ratios.append(S[i] / S[i+1])

    # Print results
    print(f"\n  Renyi Effective Rank: {er_result.renyi_effective_rank:.6f}")
    name_r, val_r, err_r = find_closest(er_result.renyi_effective_rank)
    marker = " ✓✓" if err_r < 1 else " ✓" if err_r < 5 else ""
    print(f"    → closest: {name_r} = {val_r:.6f} ({err_r:.4f}% error){marker}")

    print(f"\n  Shannon Effective Rank: {er_result.shannon_effective_rank:.6f}")
    name_s, val_s, err_s = find_closest(er_result.shannon_effective_rank)
    marker = " ✓✓" if err_s < 1 else " ✓" if err_s < 5 else ""
    print(f"    → closest: {name_s} = {val_s:.6f} ({err_s:.4f}% error){marker}")

    print(f"\n  Spectral Participation Ratio: {spectral_pr:.6f}")
    name_p, val_p, err_p = find_closest(spectral_pr)
    marker = " ✓✓" if err_p < 1 else " ✓" if err_p < 5 else ""
    print(f"    → closest: {name_p} = {val_p:.6f} ({err_p:.4f}% error){marker}")

    print(f"\n  Top Singular Values: {S[:5]}")
    print(f"\n  Singular Value Ratios:")
    significant_ratios = []
    for i, ratio in enumerate(sv_ratios[:10]):
        name_sv, val_sv, err_sv = find_closest(ratio)
        marker = " ✓✓" if err_sv < 1 else " ✓" if err_sv < 5 else ""
        if err_sv < 5:
            significant_ratios.append((f"S[{i}]/S[{i+1}]", ratio, name_sv, err_sv))
        print(f"    S[{i}]/S[{i+1}] = {ratio:.6f} → {name_sv} ({err_sv:.2f}%){marker}")

    # Energy concentration
    total_energy = np.sum(S_sq)
    cumulative = np.cumsum(S_sq) / total_energy
    n_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    n_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    n_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    print(f"\n  Energy Concentration:")
    print(f"    Components for 90% energy: {n_90}")
    print(f"    Components for 95% energy: {n_95}")
    print(f"    Components for 99% energy: {n_99}")

    # Check if these are close to constants
    for n, pct in [(n_90, 90), (n_95, 95), (n_99, 99)]:
        name_n, val_n, err_n = find_closest(n)
        if err_n < 10:
            print(f"    n_{pct} = {n} ≈ {name_n} ({err_n:.2f}% error)")

    return {
        "renyi": er_result.renyi_effective_rank,
        "shannon": er_result.shannon_effective_rank,
        "spectral_pr": spectral_pr,
        "sv_ratios": sv_ratios,
        "significant_ratios": significant_ratios,
        "n_90": n_90,
        "n_95": n_95,
        "n_99": n_99,
    }


def main():
    print("=" * 70)
    print("VRILLON FULL-RESOLUTION SPECTROGRAM ANALYSIS")
    print("=" * 70)

    from scipy.io import wavfile
    from scipy import signal
    from scipy.ndimage import zoom

    wav_path = "/tmp/vrillon_broadcast.wav"
    message_start = 10.5
    message_end = 345.0

    sample_rate, audio = wavfile.read(wav_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    start_sample = int(message_start * sample_rate)
    end_sample = int(message_end * sample_rate)
    message_audio = audio[start_sample:end_sample].astype(np.float32)
    message_audio = message_audio / np.max(np.abs(message_audio))

    print(f"\nAudio: {len(message_audio)} samples at {sample_rate} Hz")
    print(f"Duration: {len(message_audio)/sample_rate:.2f} seconds")

    # Compute full spectrogram
    print("\nComputing spectrogram...")
    frequencies, times, Sxx = signal.spectrogram(
        message_audio,
        fs=sample_rate,
        nperseg=4096,
        noverlap=2048,
    )
    Sxx_log = np.log10(Sxx + 1e-10).T  # [time × freq]
    print(f"Full spectrogram shape: {Sxx_log.shape}")

    results = {}

    # Analyze at multiple resolutions
    resolutions = [
        ("Full", Sxx_log),
    ]

    # Add downsampled versions
    for target_t, target_f in [(500, 256), (200, 100), (82, 50), (50, 25)]:
        zoom_factors = (target_t / Sxx_log.shape[0], target_f / Sxx_log.shape[1])
        downsampled = zoom(Sxx_log, zoom_factors, order=1)
        resolutions.append((f"{target_t}×{target_f}", downsampled))

    for name, matrix in resolutions:
        results[name] = analyze_spectrogram(matrix, name)

    # Summary across scales
    print("\n" + "=" * 70)
    print("SCALE-INVARIANT PATTERNS")
    print("=" * 70)

    print("\n  Renyi Rank across scales:")
    for name, r in results.items():
        name_c, _, err = find_closest(r["renyi"])
        print(f"    {name:>12}: {r['renyi']:.6f} → {name_c} ({err:.2f}% error)")

    print("\n  Spectral PR across scales:")
    for name, r in results.items():
        name_c, _, err = find_closest(r["spectral_pr"])
        print(f"    {name:>12}: {r['spectral_pr']:.6f} → {name_c} ({err:.2f}% error)")

    # Find patterns that appear across all scales
    print("\n  SV Ratios encoding constants (< 5% error) at each scale:")
    for name, r in results.items():
        if r["significant_ratios"]:
            print(f"    {name}:")
            for ratio_name, ratio_val, const, err in r["significant_ratios"][:3]:
                print(f"      {ratio_name} = {ratio_val:.4f} ≈ {const} ({err:.2f}%)")

    # Check the most precise findings
    print("\n" + "=" * 70)
    print("MOST SIGNIFICANT FINDINGS")
    print("=" * 70)

    all_findings = []
    for scale_name, r in results.items():
        # Check Renyi
        name_c, val_c, err = find_closest(r["renyi"])
        if err < 3:
            all_findings.append((scale_name, "Renyi rank", r["renyi"], name_c, err))

        # Check Spectral PR
        name_c, val_c, err = find_closest(r["spectral_pr"])
        if err < 3:
            all_findings.append((scale_name, "Spectral PR", r["spectral_pr"], name_c, err))

        # Check SV ratios
        for ratio_name, ratio_val, const, err in r["significant_ratios"]:
            if err < 3:
                all_findings.append((scale_name, ratio_name, ratio_val, const, err))

    if all_findings:
        all_findings.sort(key=lambda x: x[4])  # Sort by error
        print("\n  Constants encoded with < 3% error:")
        for scale, metric, val, const, err in all_findings[:20]:
            print(f"    [{scale:>12}] {metric:<15} = {val:.6f} ≈ {const} ({err:.4f}%)")
    else:
        print("\n  No findings with < 3% error")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
