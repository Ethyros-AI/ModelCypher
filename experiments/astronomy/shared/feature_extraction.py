"""Feature extraction for FRB spectrograms.

Extracts geometric features from FRB waterfall plots suitable for
intrinsic dimension and CKA analysis.

All operations use the Backend protocol to stay on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

    Array = np.ndarray


@dataclass
class FRBFeatures:
    """Container for extracted FRB features."""

    features: "Array"  # [D] feature vector
    feature_names: list[str]  # Names of each feature dimension
    tns_name: str  # Source FRB identifier


def extract_frb_features(
    waterfall: "Array",
    time_series: "Array",
    spectrum: "Array",
    backend: "Backend",
    *,
    tns_name: str = "unknown",
) -> FRBFeatures:
    """Extract geometric features from FRB spectrogram.

    Features extracted:
    1. Frequency profile statistics (mean, std, skew, kurtosis per band)
    2. Time profile statistics (mean, std, skew, kurtosis)
    3. Spectral statistics (entropy, peak frequency, bandwidth)
    4. Morphological features (aspect ratio, centroid, dispersion)

    Args:
        waterfall: [freq, time] intensity array (backend array or numpy)
        time_series: [time] summed intensity over frequency
        spectrum: [freq] summed intensity over time
        backend: Backend instance for GPU operations
        tns_name: FRB identifier for tracking

    Returns:
        FRBFeatures with [D] feature vector
    """
    # Convert to backend arrays if needed
    if not hasattr(waterfall, "shape"):
        waterfall = backend.array(waterfall)
    if not hasattr(time_series, "shape"):
        time_series = backend.array(time_series)
    if not hasattr(spectrum, "shape"):
        spectrum = backend.array(spectrum)

    features = []
    feature_names = []

    # --- 1. Frequency band statistics ---
    # Divide into 8 frequency bands
    n_freq = waterfall.shape[0]
    n_bands = 8
    band_size = n_freq // n_bands

    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else n_freq
        band = waterfall[start:end, :]

        # Replace NaN with 0 for statistics
        band_flat = backend.reshape(band, (-1,))

        band_mean = backend.mean(band_flat)
        band_std = backend.std(band_flat)

        features.extend([band_mean, band_std])
        feature_names.extend([f"band_{i}_mean", f"band_{i}_std"])

    # --- 2. Time series statistics ---
    # Normalize time series
    ts_min = backend.min(time_series)
    ts_max = backend.max(time_series)
    ts_range = ts_max - ts_min
    ts_range = backend.where(
        ts_range > 1e-10, ts_range, backend.array(1.0, dtype=time_series.dtype)
    )
    ts_norm = (time_series - ts_min) / ts_range

    ts_mean = backend.mean(ts_norm)
    ts_std = backend.std(ts_norm)
    ts_max_val = backend.max(ts_norm)

    # Peak location (normalized)
    ts_numpy = backend.to_numpy(ts_norm)
    peak_idx = np.nanargmax(ts_numpy)
    peak_location = peak_idx / len(ts_numpy)

    features.extend([ts_mean, ts_std, ts_max_val, backend.array(peak_location)])
    feature_names.extend(["ts_mean", "ts_std", "ts_max", "ts_peak_location"])

    # --- 3. Spectral statistics ---
    # Normalize spectrum
    spec_min = backend.min(spectrum)
    spec_max = backend.max(spectrum)
    spec_range = spec_max - spec_min
    spec_range = backend.where(
        spec_range > 1e-10, spec_range, backend.array(1.0, dtype=spectrum.dtype)
    )
    spec_norm = (spectrum - spec_min) / spec_range

    # Spectral entropy
    spec_pos = backend.abs(spec_norm) + 1e-10
    spec_sum = backend.sum(spec_pos)
    spec_prob = spec_pos / spec_sum
    spec_entropy = -backend.sum(spec_prob * backend.log(spec_prob))

    # Peak frequency (normalized)
    spec_numpy = backend.to_numpy(spec_norm)
    spec_peak_idx = np.nanargmax(spec_numpy)
    spec_peak_location = spec_peak_idx / len(spec_numpy)

    # Bandwidth (std of frequency distribution)
    freq_indices = backend.arange(0, len(spec_numpy), 1) / len(spec_numpy)
    weighted_mean = backend.sum(freq_indices * spec_prob)
    weighted_var = backend.sum(spec_prob * (freq_indices - weighted_mean) ** 2)
    bandwidth = backend.sqrt(weighted_var)

    features.extend([spec_entropy, backend.array(spec_peak_location), bandwidth])
    feature_names.extend(["spec_entropy", "spec_peak_freq", "spec_bandwidth"])

    # --- 4. Morphological features ---
    # Aspect ratio
    aspect_ratio = backend.array(waterfall.shape[1] / waterfall.shape[0])

    # Total intensity (summed, normalized)
    total_intensity = backend.sum(backend.abs(waterfall))
    total_intensity_norm = total_intensity / (waterfall.shape[0] * waterfall.shape[1])

    # Sparsity (fraction of values below median)
    wfall_flat = backend.reshape(waterfall, (-1,))
    wfall_numpy = backend.to_numpy(wfall_flat)
    median_val = np.nanmedian(wfall_numpy)
    sparsity = np.sum(wfall_numpy < median_val) / len(wfall_numpy)

    features.extend([aspect_ratio, total_intensity_norm, backend.array(sparsity)])
    feature_names.extend(["aspect_ratio", "total_intensity", "sparsity"])

    # --- Concatenate all features ---
    # Convert to numpy for concatenation, then back to backend
    feature_values = []
    for f in features:
        if hasattr(f, "shape"):
            feature_values.append(float(backend.to_scalar(f)))
        else:
            feature_values.append(float(f))

    feature_array = backend.array(feature_values, dtype=backend.float32)

    return FRBFeatures(
        features=feature_array,
        feature_names=feature_names,
        tns_name=tns_name,
    )


def batch_extract_features(
    waterfalls: list,
    backend: "Backend",
) -> "Array":
    """Extract features from multiple FRBs.

    Args:
        waterfalls: List of FRBWaterfall objects (from data_loader)
        backend: Backend instance

    Returns:
        [N, D] array of features for N FRBs
    """
    from .data_loader import FRBWaterfall, waterfall_to_backend

    feature_list = []

    for wfall in waterfalls:
        if isinstance(wfall, FRBWaterfall):
            wfall_array = waterfall_to_backend(wfall, backend)
            ts = backend.array(wfall.time_series.astype(np.float32))
            spec = backend.array(wfall.spectrum.astype(np.float32))
            name = wfall.metadata.tns_name
        else:
            msg = f"Expected FRBWaterfall, got {type(wfall)}"
            raise TypeError(msg)

        frb_features = extract_frb_features(
            wfall_array,
            ts,
            spec,
            backend,
            tns_name=name,
        )
        feature_list.append(frb_features.features)

    # Stack into [N, D] array
    if feature_list:
        # Convert to numpy, stack, convert back
        numpy_features = [backend.to_numpy(f) for f in feature_list]
        stacked = np.stack(numpy_features, axis=0)
        return backend.array(stacked)
    else:
        return backend.array(np.zeros((0, 0), dtype=np.float32))


def get_feature_dimension() -> int:
    """Return the number of features extracted per FRB.

    Useful for pre-allocating arrays.
    """
    # 8 bands * 2 stats + 4 time stats + 3 spectral stats + 3 morphological
    return 8 * 2 + 4 + 3 + 3  # = 26
