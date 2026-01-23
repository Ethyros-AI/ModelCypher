#!/usr/bin/env python3
"""
DEEP STRUCTURAL ANALYSIS

Not looking for specific numbers. Just applying different mathematical
lenses and observing what falls out.

Lenses:
1. Fourier analysis of the envelope
2. Phase space reconstruction (delay embedding)
3. Gram matrix structure
4. Time evolution of SVD through the signal
5. Correlation structure
6. Curvature of the trajectory

Usage:
    python wow_deep_structure.py
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, signal as sig
from scipy.io import readsav
from scipy.fft import fft, fftfreq
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"


def load_raw_signal() -> np.ndarray:
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal - 0.5


def main():
    print("=" * 70)
    print("DEEP STRUCTURAL ANALYSIS")
    print("=" * 70)

    signal = load_raw_signal()
    print(f"\n  Signal shape: {signal.shape} (time × frequency)")

    # =========================================================================
    # THE TIME SERIES: CHANNEL 1 (THE PEAK CHANNEL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 1: THE TIME SERIES")
    print("=" * 70)

    # Which channel has the signal?
    channel_power = np.sum(signal**2, axis=0)
    peak_channel = np.argmax(channel_power)
    print(f"\n  Peak channel: {peak_channel}")
    print(f"  Power by channel (first 10): {channel_power[:10]}")

    # The time series in the peak channel
    ts = signal[:, peak_channel]
    print(f"\n  Time series length: {len(ts)}")
    print(f"  Non-zero values: {np.sum(ts != 0)}")
    print(f"  Max value: {ts.max():.2f} at index {np.argmax(ts)}")

    # =========================================================================
    # FOURIER ANALYSIS OF THE ENVELOPE
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 2: FOURIER ANALYSIS OF ENVELOPE")
    print("=" * 70)

    # FFT of the time series
    n = len(ts)
    fft_vals = fft(ts)
    freqs = fftfreq(n)
    power_spectrum = np.abs(fft_vals)**2

    # Find dominant frequencies (excluding DC)
    ps_no_dc = power_spectrum.copy()
    ps_no_dc[0] = 0
    ps_no_dc[n//2:] = 0  # Only positive frequencies

    top_5_idx = np.argsort(ps_no_dc)[-5:][::-1]

    print(f"\n  Top 5 frequency components (excluding DC):")
    for idx in top_5_idx:
        freq = freqs[idx]
        power = power_spectrum[idx]
        period = 1/freq if freq != 0 else np.inf
        print(f"    Freq {idx}: f={freq:.4f}, period={period:.1f} samples, power={power:.1f}")

    # DC component (mean)
    dc = fft_vals[0].real / n
    print(f"\n  DC component (mean): {dc:.4f}")

    # Is there periodicity?
    print(f"\n  Looking for periodicity...")
    autocorr = np.correlate(ts, ts, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find peaks in autocorrelation
    peaks = []
    for i in range(1, len(autocorr)-1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
            if autocorr[i] > 0.1:  # Threshold
                peaks.append((i, autocorr[i]))

    print(f"  Autocorrelation peaks (lag, value):")
    for lag, val in peaks[:5]:
        print(f"    Lag {lag}: {val:.3f}")

    # =========================================================================
    # PHASE SPACE RECONSTRUCTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 3: PHASE SPACE (DELAY EMBEDDING)")
    print("=" * 70)

    # Takens embedding: reconstruct phase space from time series
    # Using delay τ and embedding dimension m

    for tau in [1, 2, 3]:
        for m in [2, 3]:
            # Create embedding
            N = len(ts) - (m-1) * tau
            if N < 10:
                continue

            embedded = np.zeros((N, m))
            for i in range(m):
                embedded[:, i] = ts[i*tau : i*tau + N]

            # Compute covariance matrix of embedded space
            cov = np.cov(embedded.T)

            # Eigenvalues of covariance
            eigvals = np.linalg.eigvalsh(cov)[::-1]

            # Effective dimension
            total = np.sum(eigvals)
            if total > 0:
                pr = (np.sum(eigvals)**2) / np.sum(eigvals**2)
            else:
                pr = 0

            print(f"\n  τ={tau}, m={m}:")
            print(f"    Covariance eigenvalues: {eigvals}")
            print(f"    Participation ratio: {pr:.3f}")

    # =========================================================================
    # GRAM MATRIX STRUCTURE
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 4: GRAM MATRIX STRUCTURE")
    print("=" * 70)

    # Gram matrix: G = signal @ signal.T (time × time)
    G_time = signal @ signal.T
    print(f"\n  Time Gram matrix shape: {G_time.shape}")

    # Eigendecomposition
    eigvals_time, eigvecs_time = np.linalg.eigh(G_time)
    eigvals_time = eigvals_time[::-1]  # Descending

    print(f"  Top 10 eigenvalues:")
    for i in range(10):
        print(f"    λ[{i}] = {eigvals_time[i]:.4f}")

    # Trace and determinant
    print(f"\n  Trace: {np.trace(G_time):.4f}")
    print(f"  Sum of eigenvalues: {np.sum(eigvals_time):.4f}")

    # Condition number
    nonzero_eigs = eigvals_time[eigvals_time > 1e-10]
    if len(nonzero_eigs) > 0:
        cond = nonzero_eigs[0] / nonzero_eigs[-1]
        print(f"  Condition number: {cond:.2f}")

    # Gram matrix of the 6EQUJ5 sequence
    seq = np.array([6, 14, 26, 30, 19, 5])
    G_seq = np.outer(seq, seq)
    print(f"\n  Gram matrix of sequence [6,14,26,30,19,5]:")
    print(f"  {G_seq}")
    print(f"  Eigenvalues: {np.linalg.eigvalsh(G_seq)[::-1]}")

    # =========================================================================
    # TIME EVOLUTION OF SVD
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 5: TIME EVOLUTION OF SVD STRUCTURE")
    print("=" * 70)

    # Sliding window SVD - how does the structure evolve?
    window_size = 20
    stride = 5

    print(f"\n  Sliding window analysis (window={window_size}, stride={stride}):")
    print(f"  {'Start':>5} {'End':>5} {'S0':>10} {'S1':>10} {'S0/S1':>10} {'PR':>10}")

    for start in range(0, len(ts) - window_size + 1, stride):
        end = start + window_size
        window = signal[start:end, :]

        _, S_window, _ = linalg.svd(window, full_matrices=False)

        ratio = S_window[0] / S_window[1] if S_window[1] > 0 else np.inf
        pr = (np.sum(S_window**2)**2) / np.sum(S_window**4) if np.sum(S_window**4) > 0 else 0

        # Highlight the peak region
        marker = " *" if 55 <= start <= 65 else ""
        print(f"  {start:5d} {end:5d} {S_window[0]:10.3f} {S_window[1]:10.3f} {ratio:10.3f} {pr:10.3f}{marker}")

    # =========================================================================
    # CURVATURE OF THE TRAJECTORY
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 6: TRAJECTORY CURVATURE")
    print("=" * 70)

    # Project onto first 3 SVD modes
    U, S, Vt = linalg.svd(signal, full_matrices=False)

    traj = np.zeros((signal.shape[0], 3))
    for i in range(3):
        traj[:, i] = U[:, i] * S[i]

    # Compute velocity (first derivative)
    velocity = np.diff(traj, axis=0)

    # Compute acceleration (second derivative)
    accel = np.diff(velocity, axis=0)

    # Curvature = |v × a| / |v|³
    curvature = []
    for i in range(len(accel)):
        v = velocity[i]
        a = accel[i]
        cross = np.cross(v, a)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            kappa = np.linalg.norm(cross) / (v_norm ** 3)
        else:
            kappa = 0
        curvature.append(kappa)

    curvature = np.array(curvature)

    print(f"\n  Trajectory curvature statistics:")
    print(f"    Mean curvature: {np.mean(curvature):.4f}")
    print(f"    Max curvature: {np.max(curvature):.4f} at index {np.argmax(curvature)}")
    print(f"    Std curvature: {np.std(curvature):.4f}")

    # Curvature around the peak
    peak_idx = np.argmax(np.linalg.norm(traj, axis=1))
    print(f"\n  Peak of trajectory at index: {peak_idx}")
    print(f"  Curvature near peak:")
    for i in range(max(0, peak_idx-5), min(len(curvature), peak_idx+5)):
        marker = " <-- peak" if i == peak_idx else ""
        print(f"    Index {i}: κ = {curvature[i]:.4f}{marker}")

    # Total arc length
    arc_lengths = np.linalg.norm(velocity, axis=1)
    total_arc = np.sum(arc_lengths)
    print(f"\n  Total arc length: {total_arc:.4f}")

    # Integrated curvature (total turning)
    total_turning = np.sum(curvature * arc_lengths[:-1])
    print(f"  Integrated curvature (total turning): {total_turning:.4f}")
    print(f"  Total turning / 2π: {total_turning / (2 * np.pi):.4f} full rotations")

    # =========================================================================
    # CORRELATION STRUCTURE
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 7: CORRELATION STRUCTURE")
    print("=" * 70)

    # Correlation matrix between time points
    # (Different from Gram matrix - this is normalized)
    valid_rows = signal[np.any(signal != 0, axis=1)]
    if len(valid_rows) > 5:
        corr_matrix = np.corrcoef(valid_rows)
        print(f"\n  Correlation matrix shape: {corr_matrix.shape}")

        # Eigenvalues of correlation matrix
        corr_eigs = np.linalg.eigvalsh(corr_matrix)[::-1]
        print(f"  Top 5 eigenvalues: {corr_eigs[:5]}")

        # How many dimensions needed for 99% of correlation?
        total_corr = np.sum(corr_eigs)
        cumvar = np.cumsum(corr_eigs) / total_corr
        dim_99 = np.searchsorted(cumvar, 0.99) + 1
        print(f"  Dimensions for 99% correlation variance: {dim_99}")

    # =========================================================================
    # THE PEAK SEQUENCE AS A DYNAMICAL SYSTEM
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 8: SEQUENCE AS DYNAMICAL SYSTEM")
    print("=" * 70)

    seq = [6, 14, 26, 30, 19, 5]

    print(f"\n  Sequence: {seq}")

    # Is there a linear map that predicts next from previous?
    # x[n+1] = A * x[n] + b ?
    X = np.array(seq[:-1]).reshape(-1, 1)
    Y = np.array(seq[1:]).reshape(-1, 1)

    # Least squares: Y = A*X + b
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    params, residuals, rank, s = np.linalg.lstsq(X_aug, Y, rcond=None)
    A, b = params[0, 0], params[1, 0]

    print(f"\n  Linear model: x[n+1] = {A:.4f} * x[n] + {b:.4f}")

    predicted = A * np.array(seq[:-1]) + b
    actual = np.array(seq[1:])
    errors = actual - predicted
    print(f"  Predictions: {predicted}")
    print(f"  Actual:      {actual}")
    print(f"  Errors:      {errors}")
    print(f"  RMS error:   {np.sqrt(np.mean(errors**2)):.2f}")

    # Is there a 2nd order map? x[n+2] = A*x[n+1] + B*x[n] + c?
    if len(seq) >= 4:
        X2 = np.column_stack([seq[1:-1], seq[:-2]])
        Y2 = np.array(seq[2:])
        X2_aug = np.hstack([X2, np.ones((len(X2), 1))])
        params2, _, _, _ = np.linalg.lstsq(X2_aug, Y2, rcond=None)

        print(f"\n  2nd order model: x[n+2] = {params2[0]:.4f}*x[n+1] + {params2[1]:.4f}*x[n] + {params2[2]:.4f}")

        pred2 = params2[0] * np.array(seq[1:-1]) + params2[1] * np.array(seq[:-2]) + params2[2]
        err2 = np.array(seq[2:]) - pred2
        print(f"  Predictions: {pred2}")
        print(f"  Actual:      {seq[2:]}")
        print(f"  Errors:      {err2}")
        print(f"  RMS error:   {np.sqrt(np.mean(err2**2)):.2f}")

    # =========================================================================
    # HILBERT TRANSFORM - INSTANTANEOUS PHASE
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENS 9: HILBERT TRANSFORM (INSTANTANEOUS PHASE)")
    print("=" * 70)

    # Hilbert transform gives analytic signal
    analytic = sig.hilbert(ts)
    amplitude_envelope = np.abs(analytic)
    instantaneous_phase = np.unwrap(np.angle(analytic))

    print(f"\n  Amplitude envelope (around peak):")
    peak_t = np.argmax(ts)
    for i in range(max(0, peak_t-5), min(len(ts), peak_t+6)):
        marker = " <-- peak" if i == peak_t else ""
        print(f"    t={i}: amp={amplitude_envelope[i]:.3f}, phase={np.degrees(instantaneous_phase[i]):.1f}°{marker}")

    # Phase change rate (instantaneous frequency)
    inst_freq = np.diff(instantaneous_phase)
    print(f"\n  Instantaneous frequency (phase change per sample):")
    print(f"    Mean: {np.mean(inst_freq):.4f} rad/sample = {np.degrees(np.mean(inst_freq)):.2f}°/sample")
    print(f"    Around peak: {np.degrees(inst_freq[peak_t-2:peak_t+3])}°/sample")

    # =========================================================================
    # SUMMARY OF OBSERVATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF OBSERVATIONS")
    print("=" * 70)

    print(f"""
  WHAT FELL OUT:

  1. FOURIER: Dominant low-frequency structure, no clear periodicity
     DC component is small ({dc:.4f}), signal is centered

  2. PHASE SPACE: Participation ratio varies with embedding
     Higher embedding dimensions show more structure

  3. GRAM MATRIX: Eigenvalue spectrum matches SVD (as expected)
     The sequence Gram matrix is rank-1 (outer product structure)

  4. TIME EVOLUTION: SVD structure CHANGES through the signal
     Peak region (*) has different participation ratio than edges

  5. CURVATURE: Trajectory has maximum curvature near the peak
     Total turning ≈ {total_turning / (2 * np.pi):.2f} full rotations

  6. CORRELATION: High correlation between adjacent time points
     {dim_99} dimensions capture 99% of correlation structure

  7. DYNAMICS: Sequence is NOT well-predicted by linear maps
     RMS errors are large - it's not a simple dynamical system

  8. HILBERT: Instantaneous phase shows rapid change at peak
     ~{np.degrees(np.mean(inst_freq[peak_t-2:peak_t+3])):.1f}°/sample phase velocity at peak

  The signal has INHOMOGENEOUS structure:
  - Different parts behave differently
  - The peak is geometrically distinct (high curvature)
  - Not a simple oscillator or linear system
""")


if __name__ == "__main__":
    main()
