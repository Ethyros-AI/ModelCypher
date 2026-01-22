"""
Audit Part 2: What does the Wow! signal actually look like?
Is the SVD structure just a reflection of having a narrowband peak?
"""

import sys
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.io import readsav

DATA_PATH = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"


def main():
    print("=" * 60)
    print("Wow! Signal Structure Audit")
    print("=" * 60)

    # Load data
    data = readsav(str(DATA_PATH))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)

    # 1. What does the signal look like in time/frequency?
    print("\n1. SIGNAL STRUCTURE")
    print("-" * 40)
    print(f"Shape: {signal.shape} (82 time steps × 50 frequency channels)")

    # Find the peak
    max_idx = np.unravel_index(np.argmax(signal), signal.shape)
    print(f"Peak location: time={max_idx[0]}, freq={max_idx[1]}")
    print(f"Peak value: {signal[max_idx]:.1f}")
    print(f"Mean background: {np.median(signal):.2f}")

    # Look at the time profile at peak frequency
    peak_freq = max_idx[1]
    time_profile = signal[:, peak_freq]
    print(f"\nTime profile at peak frequency (channel {peak_freq}):")
    print(f"  Max: {time_profile.max():.1f}")
    print(f"  Mean: {time_profile.mean():.2f}")
    print(f"  Std: {time_profile.std():.2f}")

    # Look at frequency profile at peak time
    peak_time = max_idx[0]
    freq_profile = signal[peak_time, :]
    print(f"\nFrequency profile at peak time (step {peak_time}):")
    print(f"  Max: {freq_profile.max():.1f}")
    print(f"  Mean: {freq_profile.mean():.2f}")
    print(f"  Std: {freq_profile.std():.2f}")

    # 2. Create synthetic "Wow-like" signals
    print("\n2. SYNTHETIC COMPARISON")
    print("-" * 40)
    print("Creating synthetic signals with similar structure (narrowband peak)...")

    def create_synthetic_wow(n_time=82, n_freq=50, peak_snr=30, background_mean=0.8, background_std=0.3):
        """Create a synthetic narrowband signal similar to Wow!"""
        # Background noise
        syn = np.random.randn(n_time, n_freq) * background_std + background_mean
        syn = np.maximum(syn, 0.5)  # Floor at 0.5 like real data

        # Add a narrowband peak
        peak_freq = n_freq // 2
        peak_time = n_time // 2

        # Gaussian envelope in time
        t = np.arange(n_time)
        time_envelope = np.exp(-0.5 * ((t - peak_time) / 10) ** 2)

        # Add to one frequency channel
        syn[:, peak_freq] += peak_snr * time_envelope

        return syn

    # Generate many synthetic signals and compute their SVD ratios
    n_trials = 1000
    syn_ratios_r1 = []
    syn_ratios_r2 = []

    for _ in range(n_trials):
        syn = create_synthetic_wow()
        _, S_syn, _ = linalg.svd(syn, full_matrices=False)
        syn_ratios_r1.append(S_syn[0] / S_syn[1])
        syn_ratios_r2.append(S_syn[1] / S_syn[2])

    syn_ratios_r1 = np.array(syn_ratios_r1)
    syn_ratios_r2 = np.array(syn_ratios_r2)

    # Real Wow! ratios
    _, S_real, _ = linalg.svd(signal, full_matrices=False)
    r1_real = S_real[0] / S_real[1]
    r2_real = S_real[1] / S_real[2]

    print(f"\nSynthetic 'Wow-like' signals S[0]/S[1]:")
    print(f"  Mean: {syn_ratios_r1.mean():.4f}")
    print(f"  Std: {syn_ratios_r1.std():.4f}")
    print(f"  Real Wow!: {r1_real:.4f}")
    z1 = (r1_real - syn_ratios_r1.mean()) / syn_ratios_r1.std()
    print(f"  z-score: {z1:.2f}")

    print(f"\nSynthetic 'Wow-like' signals S[1]/S[2]:")
    print(f"  Mean: {syn_ratios_r2.mean():.4f}")
    print(f"  Std: {syn_ratios_r2.std():.4f}")
    print(f"  Real Wow!: {r2_real:.4f}")
    z2 = (r2_real - syn_ratios_r2.mean()) / syn_ratios_r2.std()
    print(f"  z-score: {z2:.2f}")

    # 3. Compare to FRB control if available
    print("\n3. FRB CONTROL COMPARISON")
    print("-" * 40)

    frb_dir = Path(__file__).parent / "data" / "raw"
    if frb_dir.exists():
        import h5py
        frb_files = sorted(frb_dir.glob("FRB*.h5"))[:20]
        print(f"Found {len(frb_files)} FRB files")

        frb_ratios_r1 = []
        frb_ratios_r2 = []

        for frb_path in frb_files:
            try:
                with h5py.File(frb_path, 'r') as f:
                    wfall = f['frb']['wfall'][:]
                    wfall = wfall.astype(np.float64)
                    wfall = np.nan_to_num(wfall, nan=0.0, posinf=0.0, neginf=0.0)
                    if wfall.size > 0:
                        _, S_frb, _ = linalg.svd(wfall, full_matrices=False)
                        if len(S_frb) >= 3:
                            frb_ratios_r1.append(S_frb[0] / S_frb[1])
                            frb_ratios_r2.append(S_frb[1] / S_frb[2])
            except Exception as e:
                pass

        if frb_ratios_r1:
            frb_ratios_r1 = np.array(frb_ratios_r1)
            frb_ratios_r2 = np.array(frb_ratios_r2)

            print(f"\nFRB signals S[0]/S[1]:")
            print(f"  Mean: {frb_ratios_r1.mean():.4f}")
            print(f"  Std: {frb_ratios_r1.std():.4f}")
            print(f"  Real Wow!: {r1_real:.4f}")
            z1_frb = (r1_real - frb_ratios_r1.mean()) / frb_ratios_r1.std()
            print(f"  z-score: {z1_frb:.2f}")

            print(f"\nFRB signals S[1]/S[2]:")
            print(f"  Mean: {frb_ratios_r2.mean():.4f}")
            print(f"  Std: {frb_ratios_r2.std():.4f}")
            print(f"  Real Wow!: {r2_real:.4f}")
            z2_frb = (r2_real - frb_ratios_r2.mean()) / frb_ratios_r2.std()
            print(f"  z-score: {z2_frb:.2f}")
        else:
            print("No valid FRB data loaded")
    else:
        print(f"FRB data directory not found: {frb_dir}")

    # 4. Summary
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("\nThe key question: Are the SVD ratios unusual because of:")
    print("  A) Deep physics (φ, π encoding)")
    print("  B) Generic narrowband signal structure")
    print("  C) Something specific to the Wow! signal")
    print("\nComparing to synthetic narrowband signals tells us if (B) explains it.")


if __name__ == "__main__":
    main()
