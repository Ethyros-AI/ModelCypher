#!/usr/bin/env python3
"""Experiment 23: Validate the Information Detector.

The challenge: Can our detector distinguish signals with ACTUAL encoded
information from signals that merely LOOK structured?

If yes: High scores on astronomical signals suggest real information content.
If no: We're just detecting "structure" - meaningless without decoding.

Test cases:
1. KNOWN INFORMATION: Text encoded as radio signals (AM, FM, PSK)
2. NATURAL STRUCTURE: Pulsar signatures (structured but not "information")
3. RANDOM NOISE: Baseline for comparison
4. THE WOW! SIGNAL: Where does it fall?

The key question: Does information-bearing structure have measurable properties
that distinguish it from non-information-bearing structure?

Usage:
    poetry run python experiments/astronomy/exp23_information_validation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def compute_information_score(data_2d: np.ndarray) -> dict:
    """Compute information structure score for a 2D signal.

    Same algorithm as exp20/exp21.
    """
    if data_2d.ndim == 1:
        data_2d = data_2d.reshape(1, -1)

    time_profile = np.nanmean(data_2d, axis=0)
    freq_profile = np.nanmean(data_2d, axis=1)

    time_profile = time_profile[np.isfinite(time_profile)]
    freq_profile = freq_profile[np.isfinite(freq_profile)]

    if len(time_profile) < 3 or len(freq_profile) < 3:
        return {"score": 0.5, "interpretation": "insufficient_data", "components": {}}

    # 1. TEMPORAL ENTROPY
    if np.std(time_profile) > 1e-10:
        hist, _ = np.histogram(time_profile, bins=min(10, len(time_profile)//2), density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        temporal_entropy = -np.sum(hist * np.log2(hist)) / np.log2(len(hist))
    else:
        temporal_entropy = 1.0
    entropy_score = 1 - temporal_entropy

    # 2. BURST SHARPNESS
    if len(time_profile) > 1 and np.std(time_profile) > 1e-10:
        diffs = np.diff(time_profile)
        sharpness = np.max(np.abs(diffs)) / (np.std(time_profile) + 1e-10)
        sharpness_score = min(sharpness / 5, 1.0)
    else:
        sharpness_score = 0.0

    # 3. SPECTRAL SMOOTHNESS
    if len(freq_profile) > 1 and np.std(freq_profile) > 1e-10:
        diffs = np.diff(freq_profile)
        smoothness = 1 / (1 + np.std(diffs) / (np.std(freq_profile) + 1e-10))
        smoothness_score = smoothness
    else:
        smoothness_score = 0.5

    # 4. SIGNAL-TO-NOISE PROXY
    if np.std(time_profile) > 1e-10:
        snr_proxy = (np.max(time_profile) - np.mean(time_profile)) / np.std(time_profile)
        snr_score = min(snr_proxy / 10, 1.0)
    else:
        snr_score = 0.0

    # 5. PEAK CONCENTRATION
    if np.max(time_profile) > 0:
        normalized = time_profile / np.max(time_profile)
        concentration = 1 - (np.sum(normalized > 0.5) / len(normalized))
        concentration_score = concentration
    else:
        concentration_score = 0.5

    weights = {
        "entropy": 0.25,
        "sharpness": 0.20,
        "smoothness": 0.20,
        "snr_proxy": 0.20,
        "concentration": 0.15,
    }

    combined_score = (
        weights["entropy"] * entropy_score +
        weights["sharpness"] * sharpness_score +
        weights["smoothness"] * smoothness_score +
        weights["snr_proxy"] * snr_score +
        weights["concentration"] * concentration_score
    )

    if combined_score > 0.7:
        interpretation = "highly_structured"
    elif combined_score > 0.5:
        interpretation = "moderately_structured"
    elif combined_score > 0.3:
        interpretation = "weakly_structured"
    else:
        interpretation = "noise_like"

    return {
        "score": float(combined_score),
        "interpretation": interpretation,
        "components": {
            "entropy_score": float(entropy_score),
            "sharpness_score": float(sharpness_score),
            "smoothness_score": float(smoothness_score),
            "snr_proxy_score": float(snr_score),
            "concentration_score": float(concentration_score),
        },
    }


def generate_am_signal(message: str, shape: tuple, carrier_freq: float = 0.3) -> np.ndarray:
    """Generate AM (Amplitude Modulation) encoded signal.

    This encodes actual information in a radio-like signal.
    """
    n_freq, n_time = shape
    t = np.linspace(0, 10, n_time)

    # Convert message to binary
    binary = ''.join(format(ord(c), '08b') for c in message)

    # Create modulating signal from binary
    samples_per_bit = n_time // len(binary)
    modulating = np.zeros(n_time)
    for i, bit in enumerate(binary):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, n_time)
        modulating[start:end] = 0.3 if bit == '0' else 1.0

    # Create carrier
    carrier = np.sin(2 * np.pi * carrier_freq * n_freq * t / 10)

    # AM modulation
    am_signal = (1 + 0.5 * modulating) * carrier

    # Create 2D spectrogram-like representation
    signal_2d = np.zeros(shape)
    center_freq = n_freq // 2
    bandwidth = max(3, n_freq // 10)

    for f in range(max(0, center_freq - bandwidth), min(n_freq, center_freq + bandwidth)):
        dist = abs(f - center_freq) / bandwidth
        signal_2d[f, :] = am_signal * np.exp(-dist**2)

    # Add noise
    signal_2d += 0.1 * np.random.randn(*shape)

    return signal_2d


def generate_fm_signal(message: str, shape: tuple) -> np.ndarray:
    """Generate FM (Frequency Modulation) encoded signal.

    Information is encoded in frequency shifts.
    """
    n_freq, n_time = shape
    t = np.linspace(0, 10, n_time)

    # Convert message to binary
    binary = ''.join(format(ord(c), '08b') for c in message)

    # Create frequency shifts from binary
    samples_per_bit = n_time // len(binary)

    signal_2d = np.zeros(shape)
    center_freq = n_freq // 2
    freq_deviation = n_freq // 8

    for i, bit in enumerate(binary):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, n_time)

        # Frequency depends on bit value
        freq_idx = center_freq + freq_deviation if bit == '1' else center_freq - freq_deviation
        freq_idx = max(0, min(n_freq - 1, freq_idx))

        # Add energy at that frequency
        bandwidth = 2
        for f in range(max(0, freq_idx - bandwidth), min(n_freq, freq_idx + bandwidth)):
            signal_2d[f, start:end] = 1.0 * np.exp(-((f - freq_idx) / bandwidth)**2)

    # Add noise
    signal_2d += 0.1 * np.random.randn(*shape)

    return signal_2d


def generate_psk_signal(message: str, shape: tuple) -> np.ndarray:
    """Generate PSK (Phase Shift Keying) encoded signal.

    Information is encoded in phase transitions.
    """
    n_freq, n_time = shape
    t = np.linspace(0, 10, n_time)

    # Convert message to binary
    binary = ''.join(format(ord(c), '08b') for c in message)

    # Create phase-modulated signal
    samples_per_bit = n_time // len(binary)
    phase = np.zeros(n_time)

    for i, bit in enumerate(binary):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, n_time)
        phase[start:end] = 0 if bit == '0' else np.pi

    # Carrier with phase modulation
    carrier_freq = 5
    carrier = np.sin(2 * np.pi * carrier_freq * t + phase)

    # Create 2D representation
    signal_2d = np.zeros(shape)
    center_freq = n_freq // 2
    bandwidth = max(3, n_freq // 10)

    for f in range(max(0, center_freq - bandwidth), min(n_freq, center_freq + bandwidth)):
        dist = abs(f - center_freq) / bandwidth
        signal_2d[f, :] = carrier * np.exp(-dist**2)

    # Add noise
    signal_2d += 0.1 * np.random.randn(*shape)

    return signal_2d


def generate_pulsar_signal(shape: tuple, period: float = 0.1) -> np.ndarray:
    """Generate pulsar-like signal.

    Structured but NOT information-bearing - it's a natural phenomenon.
    Periodic pulses with dispersion (higher frequencies arrive first).
    """
    n_freq, n_time = shape
    t = np.linspace(0, 1, n_time)

    signal_2d = np.zeros(shape)

    # Create periodic pulses
    n_pulses = int(1.0 / period)
    pulse_width = int(n_time * 0.02)  # 2% duty cycle

    for pulse_num in range(n_pulses):
        pulse_center = int(pulse_num * period * n_time)

        # Dispersion: higher frequencies arrive first
        # DM delay ~ 1/f^2
        for f in range(n_freq):
            freq_factor = (f + 1) / n_freq
            delay = int((1 - freq_factor**2) * n_time * 0.05)  # 5% max delay

            pulse_time = pulse_center + delay
            if 0 <= pulse_time < n_time:
                start = max(0, pulse_time - pulse_width // 2)
                end = min(n_time, pulse_time + pulse_width // 2)

                # Gaussian pulse shape
                pulse_t = np.arange(start, end) - pulse_time
                pulse = np.exp(-pulse_t**2 / (2 * (pulse_width / 4)**2))
                signal_2d[f, start:end] = pulse

    # Add noise
    signal_2d += 0.1 * np.random.randn(*shape)

    return signal_2d


def generate_rfi_signal(shape: tuple) -> np.ndarray:
    """Generate RFI (Radio Frequency Interference) signal.

    Human-made but NOT information-bearing in this context.
    Characteristic: narrowband, persistent, often with harmonics.
    """
    n_freq, n_time = shape

    signal_2d = np.zeros(shape)

    # Primary RFI line
    rfi_freq = np.random.randint(n_freq // 4, 3 * n_freq // 4)
    signal_2d[rfi_freq, :] = 1.0

    # Harmonics
    for harmonic in [2, 3]:
        harm_freq = (rfi_freq * harmonic) % n_freq
        signal_2d[harm_freq, :] = 0.3 / harmonic

    # Time-varying intensity (60 Hz hum-like)
    hum = 0.3 * np.sin(2 * np.pi * 3 * np.linspace(0, 1, n_time))
    signal_2d[rfi_freq, :] += hum

    # Add noise
    signal_2d += 0.1 * np.random.randn(*shape)

    return signal_2d


def generate_gaussian_noise(shape: tuple) -> np.ndarray:
    """Generate pure Gaussian noise - baseline."""
    return np.random.randn(*shape)


def generate_colored_noise(shape: tuple) -> np.ndarray:
    """Generate 1/f (pink) noise - natural but unstructured."""
    n_freq, n_time = shape

    # Generate in frequency domain
    white = np.random.randn(n_freq, n_time)
    freqs = np.fft.fftfreq(n_time)
    freqs[0] = 1e-10

    # Apply 1/f filter
    fft = np.fft.fft(white, axis=1)
    pink_filter = 1 / np.sqrt(np.abs(freqs) + 0.01)
    pink = np.fft.ifft(fft * pink_filter, axis=1).real

    return pink


def run_experiment():
    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("Experiment 23: Information Detector Validation")
    print("=" * 60)
    print("\nQuestion: Can we distinguish INFORMATION from mere STRUCTURE?")

    # Standard shape for comparison
    shape = (50, 200)  # Similar to Wow! signal dimensions
    n_samples = 20

    # Test messages with varying complexity
    messages = [
        "Hi",           # Simple
        "Hello World",  # Standard
        "The quick brown fox jumps over the lazy dog",  # Complex
        "01010101",     # Binary-like
    ]

    print("\n" + "=" * 40)
    print("PART 1: KNOWN INFORMATION-BEARING SIGNALS")
    print("=" * 40)

    information_scores = {}

    # Test AM encoding
    print("\n--- AM Modulation ---")
    am_scores = []
    for msg in messages:
        scores_for_msg = []
        for _ in range(n_samples):
            signal = generate_am_signal(msg, shape)
            result = compute_information_score(signal)
            scores_for_msg.append(result["score"])
        mean_score = np.mean(scores_for_msg)
        std_score = np.std(scores_for_msg)
        am_scores.extend(scores_for_msg)
        print(f"  '{msg[:20]}...': {mean_score:.3f} ± {std_score:.3f}")

    information_scores["AM"] = {"scores": am_scores, "mean": np.mean(am_scores), "std": np.std(am_scores)}

    # Test FM encoding
    print("\n--- FM Modulation ---")
    fm_scores = []
    for msg in messages:
        scores_for_msg = []
        for _ in range(n_samples):
            signal = generate_fm_signal(msg, shape)
            result = compute_information_score(signal)
            scores_for_msg.append(result["score"])
        mean_score = np.mean(scores_for_msg)
        std_score = np.std(scores_for_msg)
        fm_scores.extend(scores_for_msg)
        print(f"  '{msg[:20]}...': {mean_score:.3f} ± {std_score:.3f}")

    information_scores["FM"] = {"scores": fm_scores, "mean": np.mean(fm_scores), "std": np.std(fm_scores)}

    # Test PSK encoding
    print("\n--- PSK Modulation ---")
    psk_scores = []
    for msg in messages:
        scores_for_msg = []
        for _ in range(n_samples):
            signal = generate_psk_signal(msg, shape)
            result = compute_information_score(signal)
            scores_for_msg.append(result["score"])
        mean_score = np.mean(scores_for_msg)
        std_score = np.std(scores_for_msg)
        psk_scores.extend(scores_for_msg)
        print(f"  '{msg[:20]}...': {mean_score:.3f} ± {std_score:.3f}")

    information_scores["PSK"] = {"scores": psk_scores, "mean": np.mean(psk_scores), "std": np.std(psk_scores)}

    print("\n" + "=" * 40)
    print("PART 2: STRUCTURED BUT NOT INFORMATION-BEARING")
    print("=" * 40)

    structure_scores = {}

    # Pulsar signals
    print("\n--- Pulsar-like Signals ---")
    pulsar_scores = []
    for _ in range(n_samples * 4):
        period = np.random.uniform(0.05, 0.2)
        signal = generate_pulsar_signal(shape, period)
        result = compute_information_score(signal)
        pulsar_scores.append(result["score"])
    print(f"  Mean: {np.mean(pulsar_scores):.3f} ± {np.std(pulsar_scores):.3f}")
    structure_scores["Pulsar"] = {"scores": pulsar_scores, "mean": np.mean(pulsar_scores), "std": np.std(pulsar_scores)}

    # RFI signals
    print("\n--- RFI Signals ---")
    rfi_scores = []
    for _ in range(n_samples * 4):
        signal = generate_rfi_signal(shape)
        result = compute_information_score(signal)
        rfi_scores.append(result["score"])
    print(f"  Mean: {np.mean(rfi_scores):.3f} ± {np.std(rfi_scores):.3f}")
    structure_scores["RFI"] = {"scores": rfi_scores, "mean": np.mean(rfi_scores), "std": np.std(rfi_scores)}

    print("\n" + "=" * 40)
    print("PART 3: NOISE BASELINES")
    print("=" * 40)

    noise_scores = {}

    # Gaussian noise
    print("\n--- Gaussian Noise ---")
    gaussian_scores = []
    for _ in range(n_samples * 4):
        signal = generate_gaussian_noise(shape)
        result = compute_information_score(signal)
        gaussian_scores.append(result["score"])
    print(f"  Mean: {np.mean(gaussian_scores):.3f} ± {np.std(gaussian_scores):.3f}")
    noise_scores["Gaussian"] = {"scores": gaussian_scores, "mean": np.mean(gaussian_scores), "std": np.std(gaussian_scores)}

    # Pink noise
    print("\n--- Pink (1/f) Noise ---")
    pink_scores = []
    for _ in range(n_samples * 4):
        signal = generate_colored_noise(shape)
        result = compute_information_score(signal)
        pink_scores.append(result["score"])
    print(f"  Mean: {np.mean(pink_scores):.3f} ± {np.std(pink_scores):.3f}")
    noise_scores["Pink"] = {"scores": pink_scores, "mean": np.mean(pink_scores), "std": np.std(pink_scores)}

    print("\n" + "=" * 40)
    print("PART 4: STATISTICAL ANALYSIS")
    print("=" * 40)

    # Combine information-bearing scores
    all_info_scores = am_scores + fm_scores + psk_scores
    all_structure_scores = pulsar_scores + rfi_scores
    all_noise_scores = gaussian_scores + pink_scores

    print("\n--- Summary Statistics ---")
    print(f"\nInformation-bearing signals (AM/FM/PSK):")
    print(f"  Mean: {np.mean(all_info_scores):.3f} ± {np.std(all_info_scores):.3f}")
    print(f"  Range: [{np.min(all_info_scores):.3f}, {np.max(all_info_scores):.3f}]")

    print(f"\nStructured but not information (Pulsar/RFI):")
    print(f"  Mean: {np.mean(all_structure_scores):.3f} ± {np.std(all_structure_scores):.3f}")
    print(f"  Range: [{np.min(all_structure_scores):.3f}, {np.max(all_structure_scores):.3f}]")

    print(f"\nNoise (Gaussian/Pink):")
    print(f"  Mean: {np.mean(all_noise_scores):.3f} ± {np.std(all_noise_scores):.3f}")
    print(f"  Range: [{np.min(all_noise_scores):.3f}, {np.max(all_noise_scores):.3f}]")

    # Statistical tests
    print("\n--- Discrimination Tests ---")

    # Information vs Structure
    t_info_struct, p_info_struct = stats.ttest_ind(all_info_scores, all_structure_scores)
    print(f"\nInformation vs Structure: t={t_info_struct:.2f}, p={p_info_struct:.4f}")

    # Information vs Noise
    t_info_noise, p_info_noise = stats.ttest_ind(all_info_scores, all_noise_scores)
    print(f"Information vs Noise: t={t_info_noise:.2f}, p={p_info_noise:.4f}")

    # Structure vs Noise
    t_struct_noise, p_struct_noise = stats.ttest_ind(all_structure_scores, all_noise_scores)
    print(f"Structure vs Noise: t={t_struct_noise:.2f}, p={p_struct_noise:.4f}")

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    d_info_struct = cohens_d(all_info_scores, all_structure_scores)
    d_info_noise = cohens_d(all_info_scores, all_noise_scores)
    d_struct_noise = cohens_d(all_structure_scores, all_noise_scores)

    print(f"\nEffect sizes (Cohen's d):")
    print(f"  Information vs Structure: d={d_info_struct:.2f}")
    print(f"  Information vs Noise: d={d_info_noise:.2f}")
    print(f"  Structure vs Noise: d={d_struct_noise:.2f}")

    print("\n" + "=" * 40)
    print("PART 5: THE WOW! SIGNAL COMPARISON")
    print("=" * 40)

    # Load Wow! signal score from exp21
    exp21_path = results_dir / "exp21_results.json"
    if exp21_path.exists():
        with open(exp21_path) as f:
            exp21_data = json.load(f)
        wow_score = exp21_data["wow_signal"]["information_score"]["score"]

        print(f"\nWow! signal score: {wow_score:.3f}")

        # Where does it fall?
        info_percentile = np.sum(np.array(all_info_scores) < wow_score) / len(all_info_scores) * 100
        struct_percentile = np.sum(np.array(all_structure_scores) < wow_score) / len(all_structure_scores) * 100
        noise_percentile = np.sum(np.array(all_noise_scores) < wow_score) / len(all_noise_scores) * 100

        print(f"\nWow! signal percentile rankings:")
        print(f"  Among information-bearing: {info_percentile:.0f}%")
        print(f"  Among structured (non-info): {struct_percentile:.0f}%")
        print(f"  Among noise: {noise_percentile:.0f}%")

        # Z-scores
        z_vs_info = (wow_score - np.mean(all_info_scores)) / np.std(all_info_scores)
        z_vs_struct = (wow_score - np.mean(all_structure_scores)) / np.std(all_structure_scores)
        z_vs_noise = (wow_score - np.mean(all_noise_scores)) / np.std(all_noise_scores)

        print(f"\nWow! signal z-scores:")
        print(f"  vs Information-bearing: {z_vs_info:.2f}σ")
        print(f"  vs Structured (non-info): {z_vs_struct:.2f}σ")
        print(f"  vs Noise: {z_vs_noise:.2f}σ")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Determine if we can distinguish information from structure
    can_distinguish_info_struct = p_info_struct < 0.05 and abs(d_info_struct) > 0.5
    can_distinguish_info_noise = p_info_noise < 0.05 and abs(d_info_noise) > 0.5
    can_distinguish_struct_noise = p_struct_noise < 0.05 and abs(d_struct_noise) > 0.5

    print(f"""
VALIDATION RESULTS:

Can we distinguish INFORMATION from STRUCTURE?
  Statistical significance: p = {p_info_struct:.4f} {'✓' if p_info_struct < 0.05 else '✗'}
  Effect size: d = {d_info_struct:.2f} {'✓' if abs(d_info_struct) > 0.5 else '✗'}
  Verdict: {'YES' if can_distinguish_info_struct else 'NO'}

Can we distinguish INFORMATION from NOISE?
  Statistical significance: p = {p_info_noise:.4f} {'✓' if p_info_noise < 0.05 else '✗'}
  Effect size: d = {d_info_noise:.2f} {'✓' if abs(d_info_noise) > 0.5 else '✗'}
  Verdict: {'YES' if can_distinguish_info_noise else 'NO'}

Can we distinguish STRUCTURE from NOISE?
  Statistical significance: p = {p_struct_noise:.4f} {'✓' if p_struct_noise < 0.05 else '✗'}
  Effect size: d = {d_struct_noise:.2f} {'✓' if abs(d_struct_noise) > 0.5 else '✗'}
  Verdict: {'YES' if can_distinguish_struct_noise else 'NO'}
""")

    if can_distinguish_info_struct:
        print("""
KEY FINDING: The detector CAN distinguish information-bearing signals
from merely structured signals.

This means: When we measure high scores on unknown signals (like the Wow!
signal), we have evidence for INFORMATION content, not just structure.
""")
    else:
        print("""
LIMITATION: The detector CANNOT reliably distinguish information-bearing
signals from merely structured signals.

This means: High scores indicate structure, but not necessarily information.
We need additional metrics to detect actual information content.
""")

    # Save results
    results = {
        "experiment": "exp23_information_validation",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(shape),
        "n_samples_per_type": n_samples,
        "information_bearing": {
            "AM": information_scores["AM"],
            "FM": information_scores["FM"],
            "PSK": information_scores["PSK"],
            "combined_mean": float(np.mean(all_info_scores)),
            "combined_std": float(np.std(all_info_scores)),
        },
        "structured_non_info": {
            "Pulsar": structure_scores["Pulsar"],
            "RFI": structure_scores["RFI"],
            "combined_mean": float(np.mean(all_structure_scores)),
            "combined_std": float(np.std(all_structure_scores)),
        },
        "noise": {
            "Gaussian": noise_scores["Gaussian"],
            "Pink": noise_scores["Pink"],
            "combined_mean": float(np.mean(all_noise_scores)),
            "combined_std": float(np.std(all_noise_scores)),
        },
        "discrimination": {
            "info_vs_struct": {
                "t_statistic": float(t_info_struct),
                "p_value": float(p_info_struct),
                "cohens_d": float(d_info_struct),
                "can_distinguish": bool(can_distinguish_info_struct),
            },
            "info_vs_noise": {
                "t_statistic": float(t_info_noise),
                "p_value": float(p_info_noise),
                "cohens_d": float(d_info_noise),
                "can_distinguish": bool(can_distinguish_info_noise),
            },
            "struct_vs_noise": {
                "t_statistic": float(t_struct_noise),
                "p_value": float(p_struct_noise),
                "cohens_d": float(d_struct_noise),
                "can_distinguish": bool(can_distinguish_struct_noise),
            },
        },
    }

    # Add Wow! comparison if available
    if exp21_path.exists():
        results["wow_signal_comparison"] = {
            "score": float(wow_score),
            "percentile_among_info": float(info_percentile),
            "percentile_among_struct": float(struct_percentile),
            "percentile_among_noise": float(noise_percentile),
            "z_vs_info": float(z_vs_info),
            "z_vs_struct": float(z_vs_struct),
            "z_vs_noise": float(z_vs_noise),
        }

    output_path = results_dir / "exp23_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
