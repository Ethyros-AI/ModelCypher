#!/usr/bin/env python3
"""Shared geometry utilities for 1977 signal analysis.

Provides common functions for loading signals, computing metrics,
and comparing to geometric constants.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Initialize backend before any domain imports
from modelcypher.backends import initialize_default_backend
initialize_default_backend()

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

# Geometric constants
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)

# All constants to test against
CONSTANTS = {
    "π": PI,
    "e": E,
    "φ": PHI,
    "√2": SQRT2,
    "√3": SQRT3,
    "π/2": PI / 2,
    "2π": 2 * PI,
    "π²": PI ** 2,
    "e²": E ** 2,
    "φ²": PHI ** 2,
    "1/π": 1 / PI,
    "1/e": 1 / E,
    "π/e": PI / E,
    "e/π": E / PI,
    "φ/π": PHI / PI,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "7": 7.0,
    "10": 10.0,
}

# Letter-to-number mapping (Big Ear encoding)
LETTER_MAP = {
    " ": 0, "": 0,
    **{str(i): i for i in range(10)},
    **{chr(ord('A') + i): 10 + i for i in range(26)},
}


class ConstantMatch(NamedTuple):
    """Result of comparing a value to geometric constants."""
    name: str
    value: float
    error_percent: float
    is_significant: bool  # < 5% error


class ManifoldMetrics(NamedTuple):
    """Complete set of manifold metrics for a signal matrix."""
    # Effective rank
    renyi_rank: float
    shannon_rank: float
    spectral_entropy: float

    # Intrinsic dimension
    intrinsic_dim_time: float | None
    intrinsic_dim_freq: float | None

    # Spectral
    spectral_pr: float
    singular_values: list[float]
    sv_ratios: list[float]

    # Geodesic
    mean_geo_euc_ratio: float | None
    geodesic_k: int | None
    curved_fraction: float | None

    # Closest constants
    renyi_match: ConstantMatch
    spectral_pr_match: ConstantMatch


def percent_error(measured: float, expected: float) -> float:
    """Calculate percent error from expected value."""
    if expected == 0:
        return float('inf')
    return abs(measured - expected) / expected * 100


def find_closest_constant(value: float, threshold: float = 5.0) -> ConstantMatch:
    """Find the geometric constant closest to the given value.

    Args:
        value: The measured value
        threshold: Error percentage below which match is "significant"

    Returns:
        ConstantMatch with name, value, error, and significance flag
    """
    best_name = "none"
    best_value = 0.0
    best_error = float("inf")

    for name, const in CONSTANTS.items():
        error = percent_error(value, const)
        if error < best_error:
            best_name = name
            best_value = const
            best_error = error

    return ConstantMatch(
        name=best_name,
        value=best_value,
        error_percent=best_error,
        is_significant=best_error < threshold,
    )


def load_wow_signal() -> np.ndarray:
    """Load the Wow! signal as an 82×50 intensity matrix."""
    data_path = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.csv"

    rows = []
    with open(data_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            intensity_values = []
            for val in parts[:50]:
                val = val.strip().upper()
                intensity_values.append(LETTER_MAP.get(val, 0))
            rows.append(intensity_values)

    return np.array(rows, dtype=np.float32)


def load_vrillon_spectrogram(
    wav_path: str = "/tmp/vrillon_broadcast.wav",
    message_start: float = 10.5,
    message_end: float = 345.0,
    target_time_bins: int = 82,
    target_freq_bins: int = 50,
    nperseg: int = 4096,
    noverlap: int = 2048,
) -> tuple[np.ndarray, float]:
    """Load Vrillon broadcast and compute spectrogram of message portion.

    Downsamples to match Wow! signal dimensions (82×50) for comparable
    manifold analysis.

    Args:
        wav_path: Path to the WAV file
        message_start: Start time of message in seconds
        message_end: End time of message in seconds
        target_time_bins: Number of time bins to downsample to (default 82 = Wow!)
        target_freq_bins: Number of freq bins to downsample to (default 50 = Wow!)
        nperseg: FFT window size
        noverlap: Overlap between windows

    Returns:
        Tuple of (spectrogram matrix [time × freq], sample_rate)
    """
    from scipy.io import wavfile
    from scipy import signal
    from scipy.ndimage import zoom

    sample_rate, audio = wavfile.read(wav_path)

    # Handle stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Extract message portion
    start_sample = int(message_start * sample_rate)
    end_sample = int(message_end * sample_rate)
    message_audio = audio[start_sample:end_sample].astype(np.float32)

    # Normalize
    message_audio = message_audio / np.max(np.abs(message_audio))

    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        message_audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Log scale for manifold analysis
    Sxx_log = np.log10(Sxx + 1e-10)

    # Transpose to [time × freq]
    Sxx_transposed = Sxx_log.T

    # Downsample to target dimensions using zoom (preserves structure better than simple resize)
    zoom_factors = (
        target_time_bins / Sxx_transposed.shape[0],
        target_freq_bins / Sxx_transposed.shape[1],
    )
    Sxx_downsampled = zoom(Sxx_transposed, zoom_factors, order=1)

    return Sxx_downsampled.astype(np.float32), sample_rate


def load_vrillon_spectrogram_full(
    wav_path: str = "/tmp/vrillon_broadcast.wav",
    message_start: float = 10.5,
    message_end: float = 345.0,
    nperseg: int = 4096,
    noverlap: int = 2048,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Load full-resolution Vrillon spectrogram (for detailed analysis).

    Returns:
        Tuple of (spectrogram [time × freq], sample_rate, frequencies, times)
    """
    from scipy.io import wavfile
    from scipy import signal

    sample_rate, audio = wavfile.read(wav_path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    start_sample = int(message_start * sample_rate)
    end_sample = int(message_end * sample_rate)
    message_audio = audio[start_sample:end_sample].astype(np.float32)
    message_audio = message_audio / np.max(np.abs(message_audio))

    frequencies, times, Sxx = signal.spectrogram(
        message_audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    Sxx_log = np.log10(Sxx + 1e-10)
    return Sxx_log.T.astype(np.float32), sample_rate, frequencies, times


def compute_spectral_participation_ratio(matrix: np.ndarray) -> tuple[float, list[float]]:
    """Compute participation ratio from singular values.

    Returns:
        Tuple of (participation_ratio, list of singular values)
    """
    _, S, _ = np.linalg.svd(matrix, full_matrices=False)
    S_sq = S ** 2
    pr = (np.sum(S_sq) ** 2) / np.sum(S_sq ** 2)
    return pr, list(S[:20])  # Return top 20 SVs


def compute_sv_ratios(singular_values: list[float], n_ratios: int = 10) -> list[float]:
    """Compute consecutive singular value ratios."""
    ratios = []
    for i in range(min(n_ratios, len(singular_values) - 1)):
        if singular_values[i + 1] > 1e-10:
            ratios.append(singular_values[i] / singular_values[i + 1])
    return ratios


def compute_geodesic_structure(matrix: np.ndarray) -> tuple[float | None, int | None, float | None]:
    """Compute geodesic vs Euclidean distance ratio.

    Returns:
        Tuple of (mean_ratio, k_neighbors, curved_fraction)
    """
    from scipy.spatial.distance import cdist

    backend = get_default_backend()
    rg = RiemannianGeometry(backend=backend)

    try:
        arr = backend.array(matrix.astype(np.float32))
        result = rg.geodesic_distances(arr, k_neighbors=None)

        backend.eval(result.distances)
        geo_dist = np.array(backend.tolist(result.distances))

        euclidean_dist = cdist(matrix, matrix)

        # Mask valid pairs
        mask = (geo_dist > 0) & (geo_dist < result.inf_value) & (euclidean_dist > 0)

        if np.sum(mask) > 0:
            ratios = geo_dist[mask] / euclidean_dist[mask]
            mean_ratio = float(np.mean(ratios))
            curved_fraction = float(np.mean(ratios > 1.01))
            return mean_ratio, result.k_neighbors, curved_fraction

    except Exception:
        pass

    return None, None, None


def compute_all_metrics(matrix: np.ndarray) -> ManifoldMetrics:
    """Compute complete manifold metrics for a signal matrix.

    Args:
        matrix: Signal matrix [time × freq]

    Returns:
        ManifoldMetrics with all measurements
    """
    backend = get_default_backend()

    # Effective rank
    er = EffectiveRank(backend=backend)
    arr = backend.array(matrix.astype(np.float32))
    er_result = er.compute(arr)

    renyi_rank = er_result.renyi_effective_rank
    shannon_rank = er_result.shannon_effective_rank
    spectral_entropy = er_result.spectral_entropy

    # Spectral participation ratio
    spectral_pr, svs = compute_spectral_participation_ratio(matrix)
    sv_ratios = compute_sv_ratios(svs)

    # Intrinsic dimension (time view and frequency view)
    id_estimator = IntrinsicDimension(backend=backend)

    intrinsic_dim_time = None
    intrinsic_dim_freq = None

    try:
        time_result = id_estimator.compute(arr, with_ci=False)
        intrinsic_dim_time = time_result.intrinsic_dimension
    except Exception:
        pass

    try:
        arr_t = backend.array(matrix.T.astype(np.float32))
        freq_result = id_estimator.compute(arr_t, with_ci=False)
        intrinsic_dim_freq = freq_result.intrinsic_dimension
    except Exception:
        pass

    # Geodesic structure
    mean_geo_euc, geo_k, curved_frac = compute_geodesic_structure(matrix)

    # Find closest constants
    renyi_match = find_closest_constant(renyi_rank)
    spectral_pr_match = find_closest_constant(spectral_pr)

    return ManifoldMetrics(
        renyi_rank=renyi_rank,
        shannon_rank=shannon_rank,
        spectral_entropy=spectral_entropy,
        intrinsic_dim_time=intrinsic_dim_time,
        intrinsic_dim_freq=intrinsic_dim_freq,
        spectral_pr=spectral_pr,
        singular_values=svs,
        sv_ratios=sv_ratios,
        mean_geo_euc_ratio=mean_geo_euc,
        geodesic_k=geo_k,
        curved_fraction=curved_frac,
        renyi_match=renyi_match,
        spectral_pr_match=spectral_pr_match,
    )


def print_metrics_report(metrics: ManifoldMetrics, title: str) -> None:
    """Print a formatted report of manifold metrics."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")

    print(f"\n  Effective Rank Analysis:")
    print(f"    Renyi rank: {metrics.renyi_rank:.6f}")
    print(f"      → closest: {metrics.renyi_match.name} = {metrics.renyi_match.value:.6f} ({metrics.renyi_match.error_percent:.4f}% error)")
    if metrics.renyi_match.is_significant:
        print(f"      ✓ SIGNIFICANT MATCH")
    print(f"    Shannon rank: {metrics.shannon_rank:.6f}")
    print(f"    Spectral entropy: {metrics.spectral_entropy:.6f}")

    print(f"\n  Spectral Participation Ratio: {metrics.spectral_pr:.6f}")
    print(f"    → closest: {metrics.spectral_pr_match.name} = {metrics.spectral_pr_match.value:.6f} ({metrics.spectral_pr_match.error_percent:.4f}% error)")
    if metrics.spectral_pr_match.is_significant:
        print(f"    ✓ SIGNIFICANT MATCH")

    if metrics.intrinsic_dim_time is not None:
        print(f"\n  Intrinsic Dimension (time view): {metrics.intrinsic_dim_time:.6f}")
        match = find_closest_constant(metrics.intrinsic_dim_time)
        print(f"    → closest: {match.name} ({match.error_percent:.4f}% error)")

    if metrics.intrinsic_dim_freq is not None:
        print(f"  Intrinsic Dimension (freq view): {metrics.intrinsic_dim_freq:.6f}")
        match = find_closest_constant(metrics.intrinsic_dim_freq)
        print(f"    → closest: {match.name} ({match.error_percent:.4f}% error)")

    if metrics.mean_geo_euc_ratio is not None:
        print(f"\n  Geodesic Structure:")
        print(f"    Mean geodesic/Euclidean ratio: {metrics.mean_geo_euc_ratio:.6f}")
        print(f"    k_neighbors for connectivity: {metrics.geodesic_k}")
        print(f"    Fraction with curvature (ratio > 1.01): {metrics.curved_fraction:.4f}")
        if metrics.mean_geo_euc_ratio > 1.0:
            print(f"    ✓ MANIFOLD HAS CURVATURE")

    print(f"\n  Top Singular Value Ratios:")
    for i, ratio in enumerate(metrics.sv_ratios[:5]):
        match = find_closest_constant(ratio)
        marker = " ✓" if match.is_significant else ""
        print(f"    S[{i}]/S[{i+1}] = {ratio:.6f} → {match.name} ({match.error_percent:.2f}% error){marker}")


def compare_metrics(m1: ManifoldMetrics, m2: ManifoldMetrics, name1: str, name2: str) -> None:
    """Print comparison of two signal metrics."""
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<30} {name1:>15} {name2:>15} {'Diff %':>10}")
    print(f"  {'-' * 70}")

    # Renyi rank
    diff = percent_error(m1.renyi_rank, m2.renyi_rank)
    print(f"  {'Renyi Rank':<30} {m1.renyi_rank:>15.6f} {m2.renyi_rank:>15.6f} {diff:>10.2f}%")

    # Spectral PR
    diff = percent_error(m1.spectral_pr, m2.spectral_pr)
    print(f"  {'Spectral PR':<30} {m1.spectral_pr:>15.6f} {m2.spectral_pr:>15.6f} {diff:>10.2f}%")

    # Intrinsic dim time
    if m1.intrinsic_dim_time and m2.intrinsic_dim_time:
        diff = percent_error(m1.intrinsic_dim_time, m2.intrinsic_dim_time)
        print(f"  {'Intrinsic Dim (time)':<30} {m1.intrinsic_dim_time:>15.6f} {m2.intrinsic_dim_time:>15.6f} {diff:>10.2f}%")

    # Geodesic ratio
    if m1.mean_geo_euc_ratio and m2.mean_geo_euc_ratio:
        diff = percent_error(m1.mean_geo_euc_ratio, m2.mean_geo_euc_ratio)
        print(f"  {'Mean Geo/Euc Ratio':<30} {m1.mean_geo_euc_ratio:>15.6f} {m2.mean_geo_euc_ratio:>15.6f} {diff:>10.2f}%")


if __name__ == "__main__":
    # Quick test
    print("Loading Wow! signal...")
    wow = load_wow_signal()
    print(f"Shape: {wow.shape}")

    print("\nComputing metrics...")
    metrics = compute_all_metrics(wow)
    print_metrics_report(metrics, "WOW! SIGNAL MANIFOLD METRICS")
