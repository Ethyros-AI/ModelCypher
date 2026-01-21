"""Data loading utilities for CHIME FRB data.

Loads FRB waterfall plots (spectrograms) from HDF5 files and converts
them to backend arrays for geometric analysis.

Data format reference: https://chime-frb-open-data.github.io/waterfall/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass
class FRBMetadata:
    """Metadata for a single FRB detection."""

    tns_name: str  # TNS identifier (e.g., "FRB20180725A")
    dm: float  # Dispersion measure (pc/cm³)
    snr: float | None  # Signal-to-noise ratio (if computed)
    freq_range: tuple[float, float]  # MHz
    time_range: tuple[float, float]  # ms relative to peak
    n_freq_channels: int
    n_time_bins: int


@dataclass
class FRBWaterfall:
    """Container for FRB waterfall data and metadata."""

    waterfall: "np.ndarray"  # [freq, time] intensity array
    metadata: FRBMetadata
    time_axis: "np.ndarray"  # Time values in ms
    freq_axis: "np.ndarray"  # Frequency values in MHz
    time_series: "np.ndarray"  # Sum over frequency
    spectrum: "np.ndarray"  # Sum over time


def load_frb_waterfall(
    path: str | Path,
    bin_freq_factor: int = 16,
    remove_rfi: bool = True,
) -> FRBWaterfall:
    """Load a single FRB waterfall from HDF5 file.

    Args:
        path: Path to HDF5 file (e.g., FRB20180725A_waterfall.h5)
        bin_freq_factor: Factor to bin frequency channels (default 16 reduces
            16384 channels to 1024)
        remove_rfi: Whether to mask high-variance RFI channels

    Returns:
        FRBWaterfall containing the data and metadata
    """
    path = Path(path)
    if not path.exists():
        msg = f"FRB file not found: {path}"
        raise FileNotFoundError(msg)

    with h5py.File(path, "r") as f:
        data = f["frb"]

        # Extract metadata
        tns_name = data.attrs["tns_name"]
        if isinstance(tns_name, bytes):
            tns_name = tns_name.decode()

        dm = float(data.attrs["dm"][()])

        # Extract arrays
        wfall = data["wfall"][:]
        plot_time = data["plot_time"][:]
        plot_freq = data["plot_freq"][:]
        ts = data["ts"][:]
        spec = data["spec"][:]
        extent = data["extent"][:]

    # RFI removal (mask high-variance channels)
    if remove_rfi:
        wfall = _remove_rfi(wfall, spec)

    # Bin frequency channels
    if bin_freq_factor > 1:
        wfall = _bin_freq_channels(wfall, bin_freq_factor)
        plot_freq = _bin_freq_channels(plot_freq.reshape(-1, 1), bin_freq_factor).flatten()

    # Recompute time series after RFI masking
    ts = np.nansum(wfall, axis=0)

    # Compute SNR via boxcar convolution
    snr = _compute_snr(ts)

    metadata = FRBMetadata(
        tns_name=tns_name,
        dm=dm,
        snr=snr,
        freq_range=(float(extent[2]), float(extent[3])),
        time_range=(float(extent[0]), float(extent[1])),
        n_freq_channels=wfall.shape[0],
        n_time_bins=wfall.shape[1],
    )

    return FRBWaterfall(
        waterfall=wfall,
        metadata=metadata,
        time_axis=plot_time,
        freq_axis=plot_freq,
        time_series=ts,
        spectrum=spec,
    )


def load_frb_batch(
    paths: list[str | Path],
    bin_freq_factor: int = 16,
    remove_rfi: bool = True,
) -> list[FRBWaterfall]:
    """Load multiple FRB waterfalls.

    Args:
        paths: List of paths to HDF5 files
        bin_freq_factor: Factor to bin frequency channels
        remove_rfi: Whether to mask RFI channels

    Returns:
        List of FRBWaterfall objects
    """
    waterfalls = []
    for path in paths:
        try:
            wfall = load_frb_waterfall(path, bin_freq_factor, remove_rfi)
            waterfalls.append(wfall)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    return waterfalls


def get_frb_metadata(path: str | Path) -> FRBMetadata:
    """Extract only metadata from an FRB file (faster than full load)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        data = f["frb"]
        tns_name = data.attrs["tns_name"]
        if isinstance(tns_name, bytes):
            tns_name = tns_name.decode()

        dm = float(data.attrs["dm"][()])
        extent = data["extent"][:]
        wfall_shape = data["wfall"].shape

    return FRBMetadata(
        tns_name=tns_name,
        dm=dm,
        snr=None,  # Not computed without full load
        freq_range=(float(extent[2]), float(extent[3])),
        time_range=(float(extent[0]), float(extent[1])),
        n_freq_channels=wfall_shape[0],
        n_time_bins=wfall_shape[1],
    )


def waterfall_to_backend(wfall: FRBWaterfall, backend: "Backend") -> "np.ndarray":
    """Convert waterfall to backend array.

    Replaces NaN values with median and converts to backend tensor.
    """
    data = wfall.waterfall.copy()
    data[np.isnan(data)] = np.nanmedian(data)
    return backend.array(data.astype(np.float32))


# --- Private helpers ---


def _remove_rfi(wfall: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Mask RFI channels with high variance.

    Based on CHIME/FRB Open Data tutorial:
    https://chime-frb-open-data.github.io/waterfall/
    """
    q1 = np.nanquantile(spec, 0.25)
    q3 = np.nanquantile(spec, 0.75)
    iqr = q3 - q1

    rfi_masking_var_factor = 3
    channel_variance = np.nanvar(wfall, axis=1)
    mean_channel_variance = np.nanmean(channel_variance)

    with np.errstate(invalid="ignore"):
        rfi_mask = (
            (channel_variance > rfi_masking_var_factor * mean_channel_variance)
            | (spec[::-1] < q1 - 1.5 * iqr)
            | (spec[::-1] > q3 + 1.5 * iqr)
        )

    wfall = wfall.copy()
    wfall[rfi_mask, ...] = np.nan
    return wfall


def _bin_freq_channels(data: np.ndarray, fbin_factor: int = 4) -> np.ndarray:
    """Bin frequency channels by averaging adjacent channels."""
    num_chan = data.shape[0]
    if num_chan % fbin_factor != 0:
        # Truncate to nearest multiple
        num_chan = (num_chan // fbin_factor) * fbin_factor
        data = data[:num_chan]

    new_shape = (num_chan // fbin_factor, fbin_factor) + data.shape[1:]
    return np.nanmean(data.reshape(new_shape), axis=1)


def _compute_snr(ts: np.ndarray, min_width: int = 1, max_width: int = 128) -> float:
    """Compute SNR via boxcar convolution.

    Finds optimal boxcar width and returns peak SNR.
    """
    max_width = min(max_width, len(ts) - 2)
    widths = range(min_width, max_width + 1)

    best_snr = 0.0
    for width in widths:
        kernel = np.ones(width, dtype=np.float32) / np.sqrt(width)
        convolved = np.convolve(ts, kernel, mode="same")
        peak_snr = np.nanmax(convolved)
        if peak_snr > best_snr:
            best_snr = peak_snr

    return float(best_snr)
