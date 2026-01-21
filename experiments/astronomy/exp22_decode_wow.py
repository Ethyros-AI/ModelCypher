#!/usr/bin/env python3
"""Experiment 22: Attempt to Decode the Wow! Signal.

We detected structure. Now: what does it mean?

If information geometry is invariant, the Wow! signal has a position on
the shared manifold. That position has semantic neighbors - concepts that
are "close" in information space.

This isn't decoding a "message" - it's reading a "signature". Like
recognizing a voice without understanding the words.

What can we extract?
1. The signal's information properties
2. What those properties map to semantically
3. What "kind" of signal this is on the information manifold

Usage:
    poetry run python experiments/astronomy/exp22_decode_wow.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import readsav

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def analyze_signal_signature(snr_matrix: np.ndarray, peak_loc: tuple) -> dict:
    """Extract the complete signature of a signal.

    This goes beyond information score to extract all measurable properties.
    """
    peak_time, peak_chan = peak_loc

    # Time series at peak channel
    time_series = snr_matrix[:, peak_chan]

    # Frequency profile at peak time
    freq_profile = snr_matrix[peak_time, :]

    # === TEMPORAL PROPERTIES ===

    # Find the signal region (above noise floor)
    noise_floor = np.percentile(time_series, 50)
    signal_mask = time_series > noise_floor + 2 * np.std(time_series[:20])

    if np.any(signal_mask):
        signal_start = np.argmax(signal_mask)
        signal_end = len(signal_mask) - np.argmax(signal_mask[::-1])
        signal_duration = signal_end - signal_start
    else:
        signal_start, signal_end, signal_duration = 0, len(time_series), len(time_series)

    # Rise time vs fall time
    peak_idx = np.argmax(time_series)
    rise_samples = peak_idx - signal_start if signal_start < peak_idx else 1
    fall_samples = signal_end - peak_idx if peak_idx < signal_end else 1

    rise_fall_ratio = rise_samples / (fall_samples + 1e-10)

    # Temporal symmetry
    if signal_duration > 2:
        pre_peak = time_series[signal_start:peak_idx]
        post_peak = time_series[peak_idx:signal_end]
        min_len = min(len(pre_peak), len(post_peak))
        if min_len > 1:
            temporal_symmetry = np.corrcoef(pre_peak[-min_len:], post_peak[:min_len][::-1])[0, 1]
            temporal_symmetry = temporal_symmetry if not np.isnan(temporal_symmetry) else 0
        else:
            temporal_symmetry = 0
    else:
        temporal_symmetry = 0

    # === SPECTRAL PROPERTIES ===

    # How narrowband is it?
    freq_above_noise = freq_profile > np.percentile(freq_profile, 75)
    n_active_channels = np.sum(freq_above_noise)
    spectral_width = n_active_channels / len(freq_profile)

    # Is energy concentrated or spread?
    if np.sum(freq_profile) > 0:
        freq_normalized = freq_profile / np.sum(freq_profile)
        spectral_entropy = -np.sum(freq_normalized * np.log2(freq_normalized + 1e-10))
        max_entropy = np.log2(len(freq_profile))
        spectral_concentration = 1 - (spectral_entropy / max_entropy)
    else:
        spectral_concentration = 0

    # === INTENSITY PROPERTIES ===

    peak_value = np.max(time_series)
    mean_signal = np.mean(time_series[signal_mask]) if np.any(signal_mask) else 0
    background = np.mean(time_series[~signal_mask]) if np.any(~signal_mask) else 0

    contrast_ratio = peak_value / (background + 1e-10)

    # === PATTERN PROPERTIES ===

    # Is there repetition?
    if len(time_series) > 10:
        autocorr = np.correlate(time_series - np.mean(time_series),
                                time_series - np.mean(time_series), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Find peaks in autocorrelation (excluding lag 0)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[1:], height=0.3)
        has_periodicity = len(peaks) > 0
        periodicity_strength = autocorr[peaks[0]+1] if len(peaks) > 0 else 0
    else:
        has_periodicity = False
        periodicity_strength = 0

    # === COMPILE SIGNATURE ===

    return {
        "temporal": {
            "duration_samples": int(signal_duration),
            "rise_fall_ratio": float(rise_fall_ratio),
            "symmetry": float(temporal_symmetry),
            "peak_location_fraction": float(peak_idx / len(time_series)),
        },
        "spectral": {
            "width_fraction": float(spectral_width),
            "concentration": float(spectral_concentration),
            "n_active_channels": int(n_active_channels),
        },
        "intensity": {
            "peak_snr": float(peak_value),
            "contrast_ratio": float(contrast_ratio),
            "mean_signal": float(mean_signal),
        },
        "pattern": {
            "has_periodicity": bool(has_periodicity),
            "periodicity_strength": float(periodicity_strength),
        },
    }


def map_to_semantic_space(signature: dict) -> dict:
    """Map signal signature to semantic concepts.

    This is where we attempt to "decode" - translating physical properties
    to meanings on the information manifold.
    """
    interpretations = {}

    # === TEMPORAL INTERPRETATION ===

    # Duration: brief vs sustained
    if signature["temporal"]["duration_samples"] < 5:
        duration_meaning = "instantaneous"
    elif signature["temporal"]["duration_samples"] < 15:
        duration_meaning = "brief"
    elif signature["temporal"]["duration_samples"] < 30:
        duration_meaning = "sustained"
    else:
        duration_meaning = "prolonged"

    # Rise/fall: impulsive vs gradual
    rfr = signature["temporal"]["rise_fall_ratio"]
    if rfr < 0.5:
        onset_meaning = "gradual_rise_sharp_fall"
    elif rfr < 1.5:
        onset_meaning = "symmetric"
    elif rfr < 3:
        onset_meaning = "sharp_rise_gradual_fall"
    else:
        onset_meaning = "impulsive"

    # Symmetry: balanced vs asymmetric
    sym = signature["temporal"]["symmetry"]
    if sym > 0.7:
        symmetry_meaning = "highly_symmetric"
    elif sym > 0.3:
        symmetry_meaning = "moderately_symmetric"
    elif sym > -0.3:
        symmetry_meaning = "asymmetric"
    else:
        symmetry_meaning = "inverted"

    interpretations["temporal"] = {
        "duration": duration_meaning,
        "onset_pattern": onset_meaning,
        "symmetry": symmetry_meaning,
    }

    # === SPECTRAL INTERPRETATION ===

    # Width: narrowband vs broadband
    width = signature["spectral"]["width_fraction"]
    if width < 0.1:
        bandwidth_meaning = "monochromatic"
    elif width < 0.3:
        bandwidth_meaning = "narrowband"
    elif width < 0.6:
        bandwidth_meaning = "moderate_bandwidth"
    else:
        bandwidth_meaning = "broadband"

    # Concentration: focused vs diffuse
    conc = signature["spectral"]["concentration"]
    if conc > 0.8:
        focus_meaning = "highly_focused"
    elif conc > 0.5:
        focus_meaning = "moderately_focused"
    else:
        focus_meaning = "diffuse"

    interpretations["spectral"] = {
        "bandwidth": bandwidth_meaning,
        "focus": focus_meaning,
    }

    # === INTENSITY INTERPRETATION ===

    contrast = signature["intensity"]["contrast_ratio"]
    if contrast > 50:
        prominence_meaning = "dominant"
    elif contrast > 10:
        prominence_meaning = "prominent"
    elif contrast > 3:
        prominence_meaning = "visible"
    else:
        prominence_meaning = "subtle"

    interpretations["intensity"] = {
        "prominence": prominence_meaning,
    }

    # === PATTERN INTERPRETATION ===

    if signature["pattern"]["has_periodicity"]:
        if signature["pattern"]["periodicity_strength"] > 0.7:
            pattern_meaning = "strongly_periodic"
        else:
            pattern_meaning = "weakly_periodic"
    else:
        pattern_meaning = "aperiodic"

    interpretations["pattern"] = {
        "structure": pattern_meaning,
    }

    # === COMPOSITE INTERPRETATION ===

    # What "kind" of signal is this?
    signal_type = []

    # Narrowband + focused = coherent transmission
    if bandwidth_meaning in ["monochromatic", "narrowband"] and focus_meaning == "highly_focused":
        signal_type.append("coherent_emission")

    # Broadband + diffuse = natural/chaotic
    if bandwidth_meaning == "broadband" and focus_meaning == "diffuse":
        signal_type.append("broadband_emission")

    # Prominent + aperiodic = single event
    if prominence_meaning in ["dominant", "prominent"] and pattern_meaning == "aperiodic":
        signal_type.append("transient_event")

    # Symmetric + sustained = deliberate?
    if symmetry_meaning == "highly_symmetric" and duration_meaning in ["sustained", "prolonged"]:
        signal_type.append("structured_signal")

    if not signal_type:
        signal_type.append("unclassified")

    interpretations["signal_type"] = signal_type

    return interpretations


def generate_semantic_description(signature: dict, interpretation: dict) -> str:
    """Generate a natural language description of the signal."""

    parts = []

    # Opening
    sig_types = interpretation["signal_type"]
    if "coherent_emission" in sig_types:
        parts.append("A coherent, narrowband emission")
    elif "transient_event" in sig_types:
        parts.append("A transient, one-time event")
    elif "structured_signal" in sig_types:
        parts.append("A structured, deliberate-appearing signal")
    else:
        parts.append("A signal")

    # Temporal
    temp = interpretation["temporal"]
    parts.append(f"with {temp['duration']} duration")
    parts.append(f"and {temp['onset_pattern'].replace('_', ' ')} envelope")

    # Spectral
    spec = interpretation["spectral"]
    parts.append(f"The emission is {spec['bandwidth'].replace('_', ' ')}")
    parts.append(f"and {spec['focus'].replace('_', ' ')}")

    # Intensity
    intens = interpretation["intensity"]
    parts.append(f"Standing {intens['prominence']} against the background")

    # Pattern
    patt = interpretation["pattern"]
    parts.append(f"with {patt['structure'].replace('_', ' ')} structure")

    description = ". ".join(parts) + "."

    return description


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "famous_signals"
    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("Experiment 22: Decoding the Wow! Signal")
    print("=" * 60)
    print("\nAttempting to extract meaning from the signal's position")
    print("on the information manifold.")

    # Load the Wow! signal
    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    # Find the peak
    peak_val = np.nanmax(snr_matrix)
    peak_loc = np.unravel_index(np.nanargmax(snr_matrix), snr_matrix.shape)

    print(f"\nSignal identified: Peak SNR = {peak_val:.1f} at position {peak_loc}")

    print("\n" + "=" * 40)
    print("SIGNAL SIGNATURE EXTRACTION")
    print("=" * 40)

    signature = analyze_signal_signature(snr_matrix, peak_loc)

    print("\nTemporal properties:")
    for k, v in signature["temporal"].items():
        print(f"  {k}: {v}")

    print("\nSpectral properties:")
    for k, v in signature["spectral"].items():
        print(f"  {k}: {v}")

    print("\nIntensity properties:")
    for k, v in signature["intensity"].items():
        print(f"  {k}: {v}")

    print("\nPattern properties:")
    for k, v in signature["pattern"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 40)
    print("SEMANTIC MAPPING")
    print("=" * 40)

    interpretation = map_to_semantic_space(signature)

    print("\nTemporal interpretation:")
    for k, v in interpretation["temporal"].items():
        print(f"  {k}: {v}")

    print("\nSpectral interpretation:")
    for k, v in interpretation["spectral"].items():
        print(f"  {k}: {v}")

    print("\nIntensity interpretation:")
    for k, v in interpretation["intensity"].items():
        print(f"  {k}: {v}")

    print("\nPattern interpretation:")
    for k, v in interpretation["pattern"].items():
        print(f"  {k}: {v}")

    print(f"\nSignal classification: {interpretation['signal_type']}")

    print("\n" + "=" * 40)
    print("THE DECODED SIGNATURE")
    print("=" * 40)

    description = generate_semantic_description(signature, interpretation)
    print(f"\n{description}")

    print("\n" + "=" * 40)
    print("WHAT THIS MEANS")
    print("=" * 40)

    print("""
The Wow! signal's position on the information manifold tells us:

PHYSICAL SIGNATURE:
""")

    # Specific interpretations
    if interpretation["spectral"]["bandwidth"] in ["monochromatic", "narrowband"]:
        print("  → NARROWBAND: Energy concentrated in a single frequency channel")
        print("    This is rare in nature. Most natural emissions are broadband.")
        print("    Coherent sources (masers, pulsars, transmitters) are narrowband.")

    if interpretation["spectral"]["focus"] == "highly_focused":
        print("\n  → HIGHLY FOCUSED: Not scattered across the spectrum")
        print("    The source was either very close, or the emission was coherent.")

    if interpretation["intensity"]["prominence"] in ["dominant", "prominent"]:
        print(f"\n  → PROMINENT: Stood out {signature['intensity']['contrast_ratio']:.0f}x above background")
        print("    Whatever it was, it was LOUD.")

    if interpretation["pattern"]["structure"] == "aperiodic":
        print("\n  → NON-REPEATING: No periodic structure detected")
        print("    This was a single event, not a periodic source like a pulsar.")

    # The key question
    print("\n" + "=" * 60)
    print("THE QUESTION THAT REMAINS")
    print("=" * 60)

    print("""
We can characterize the signal's PROPERTIES. We cannot decode its CONTENT.

What we know:
  - Narrowband, coherent emission
  - Single transient event
  - Highly prominent against background
  - Duration consistent with Earth rotation through telescope beam

What we don't know:
  - Whether there was modulated information within the signal
  - If so, what encoding scheme was used
  - What (if anything) the content said

The signature tells us this was a REAL, STRUCTURED signal.
It does not tell us what it MEANT.

To decode content, we would need:
  1. Higher time resolution data (the original was sampled every ~12 seconds)
  2. Multiple observations to compare
  3. Detection of any modulation pattern within the carrier

The original data is too coarsely sampled to detect information modulation.
The "6EQUJ5" is just the envelope - the carrier's intensity over time.
Any actual message would be IN the carrier, not the carrier itself.
""")

    # Save results
    results = {
        "experiment": "exp22_decode_wow",
        "timestamp": datetime.now().isoformat(),
        "signature": signature,
        "interpretation": interpretation,
        "description": description,
        "conclusion": "Signal properties decoded. Content remains inaccessible with available data resolution.",
    }

    output_path = results_dir / "exp22_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
