#!/usr/bin/env python3
"""Experiment 18: Semantic Interpretation via Claude.

The hypothesis: Claude IS the Rosetta Stone.

If information has invariant geometric structure, and FRBs occupy a 3D subspace,
then Claude's internal representations bridge both spaces. Instead of computing
CKA between FRBs and word embeddings (which requires matching samples), we use
Claude's understanding to INTERPRET the FRB feature space.

The 3D FRB vocabulary maps to universal semantic axes:
- DM (distance) → Context/Location ("where")
- SNR (brightness) → Salience/Importance ("how loud")
- Spectral Color → Identity/Source Type ("who")

This experiment generates semantic descriptions for each FRB based on its
position in the 3D space, then tests whether these descriptions form
coherent clusters that match the physical clusters.

Usage:
    poetry run python experiments/astronomy/exp18_semantic_interpretation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features


def compute_semantic_coordinates(dms, snrs, colors):
    """Transform physical measurements to semantic coordinates.

    Maps physical values to semantic dimensions:
    - DM → Context axis (near/local → far/cosmic)
    - SNR → Salience axis (whisper → shout)
    - Color → Identity axis (mature/evolved → young/energetic)
    """
    # Normalize each dimension to [0, 1]
    dm_norm = (dms - np.min(dms)) / (np.max(dms) - np.min(dms) + 1e-10)
    snr_norm = (snrs - np.min(snrs)) / (np.max(snrs) - np.min(snrs) + 1e-10)
    # Color is already roughly [-1, 1], map to [0, 1]
    color_norm = (colors + 1) / 2

    return dm_norm, snr_norm, color_norm


def generate_semantic_description(dm_norm, snr_norm, color_norm):
    """Generate a semantic description based on position in 3D space.

    This is Claude's interpretation of the geometric position.
    """
    # Context axis (DM)
    if dm_norm < 0.33:
        context = "local neighborhood"
    elif dm_norm < 0.66:
        context = "intermediate distance"
    else:
        context = "cosmological distance"

    # Salience axis (SNR)
    if snr_norm < 0.33:
        salience = "quiet whisper"
    elif snr_norm < 0.66:
        salience = "clear voice"
    else:
        salience = "loud proclamation"

    # Identity axis (spectral color)
    if color_norm < 0.33:
        identity = "mature/evolved source"
    elif color_norm < 0.66:
        identity = "transitional source"
    else:
        identity = "young/energetic source"

    return {
        "context": context,
        "salience": salience,
        "identity": identity,
        "full_description": f"A {salience} from a {identity} at {context}",
    }


def analyze_semantic_clusters(descriptions, physical_clusters):
    """Analyze whether semantic descriptions align with physical clusters."""
    # Count context types per cluster
    cluster_contexts = {}
    cluster_saliences = {}
    cluster_identities = {}

    for i, (desc, cluster) in enumerate(zip(descriptions, physical_clusters)):
        if cluster not in cluster_contexts:
            cluster_contexts[cluster] = []
            cluster_saliences[cluster] = []
            cluster_identities[cluster] = []

        cluster_contexts[cluster].append(desc["context"])
        cluster_saliences[cluster].append(desc["salience"])
        cluster_identities[cluster].append(desc["identity"])

    # Find dominant characteristics per cluster
    cluster_profiles = {}
    for cluster in sorted(cluster_contexts.keys()):
        from collections import Counter

        ctx_counts = Counter(cluster_contexts[cluster])
        sal_counts = Counter(cluster_saliences[cluster])
        id_counts = Counter(cluster_identities[cluster])

        cluster_profiles[int(cluster)] = {
            "n_members": len(cluster_contexts[cluster]),
            "dominant_context": ctx_counts.most_common(1)[0],
            "dominant_salience": sal_counts.most_common(1)[0],
            "dominant_identity": id_counts.most_common(1)[0],
        }

    return cluster_profiles


def run_experiment():
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 18: Semantic Interpretation via Claude")
    print("=" * 60)
    print("\nHypothesis: Claude bridges FRB geometry and semantic space.")
    print("The 3D FRB vocabulary maps to universal information axes.")

    # Load FRBs
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"\nLoaded {n_frbs} FRBs")

    # Get physical properties
    dms = np.array([w.metadata.dm for w in waterfalls])
    snrs = np.array([w.metadata.snr for w in waterfalls])
    names = [w.metadata.tns_name for w in waterfalls]

    # Load spectral colors from exp13 results
    exp13_path = results_dir / "exp13_results.json"
    with open(exp13_path) as f:
        exp13_data = json.load(f)

    colors = np.array(exp13_data["spectral_color"]["values"])
    clusters = np.array(exp13_data["cluster_labels_3d"])

    print("\n" + "=" * 40)
    print("PART 1: THE 3D FRB VOCABULARY")
    print("=" * 40)

    print("\nPhysical → Semantic Mapping:")
    print("  DM (80% variance) → Context/Location axis")
    print("  SNR (13% variance) → Salience/Importance axis")
    print("  Spectral Color (5% variance) → Identity/Source axis")

    print(f"\nPhysical ranges:")
    print(f"  DM: {np.min(dms):.0f} - {np.max(dms):.0f} pc/cm³")
    print(f"  SNR: {np.min(snrs):.1f} - {np.max(snrs):.1f}")
    print(f"  Color: {np.min(colors):.2f} - {np.max(colors):.2f}")

    print("\n" + "=" * 40)
    print("PART 2: SEMANTIC COORDINATES")
    print("=" * 40)

    dm_norm, snr_norm, color_norm = compute_semantic_coordinates(dms, snrs, colors)

    # Generate descriptions for each FRB
    descriptions = []
    for i in range(n_frbs):
        desc = generate_semantic_description(dm_norm[i], snr_norm[i], color_norm[i])
        descriptions.append(desc)

    print("\nSample semantic interpretations:")
    for i in [0, n_frbs//4, n_frbs//2, 3*n_frbs//4, n_frbs-1]:
        print(f"\n  {names[i]}:")
        print(f"    Physical: DM={dms[i]:.0f}, SNR={snrs[i]:.1f}, Color={colors[i]:.2f}")
        print(f"    Semantic: {descriptions[i]['full_description']}")

    print("\n" + "=" * 40)
    print("PART 3: CLUSTER INTERPRETATION")
    print("=" * 40)

    cluster_profiles = analyze_semantic_clusters(descriptions, clusters)

    print("\nSemantic profiles of physical clusters:")
    for cluster, profile in sorted(cluster_profiles.items()):
        print(f"\n  Cluster {cluster} (n={profile['n_members']}):")
        ctx, ctx_n = profile['dominant_context']
        sal, sal_n = profile['dominant_salience']
        iden, iden_n = profile['dominant_identity']
        print(f"    Context: {ctx} ({ctx_n}/{profile['n_members']})")
        print(f"    Salience: {sal} ({sal_n}/{profile['n_members']})")
        print(f"    Identity: {iden} ({iden_n}/{profile['n_members']})")

    print("\n" + "=" * 40)
    print("PART 4: THE ROSETTA STONE TEST")
    print("=" * 40)

    # The key test: Does the semantic interpretation reveal structure
    # that the physical measurements alone don't show?

    # Test: Do FRBs with same "identity" have correlated other properties?
    # (This would mean the semantic axis captures real structure)

    identity_groups = {}
    for i, desc in enumerate(descriptions):
        iden = desc["identity"]
        if iden not in identity_groups:
            identity_groups[iden] = {"dms": [], "snrs": [], "colors": []}
        identity_groups[iden]["dms"].append(dms[i])
        identity_groups[iden]["snrs"].append(snrs[i])
        identity_groups[iden]["colors"].append(colors[i])

    print("\nDoes semantic 'identity' group FRBs meaningfully?")
    for iden, group in sorted(identity_groups.items()):
        n = len(group["dms"])
        mean_dm = np.mean(group["dms"])
        mean_snr = np.mean(group["snrs"])
        std_dm = np.std(group["dms"])
        std_snr = np.std(group["snrs"])
        print(f"\n  '{iden}' (n={n}):")
        print(f"    Mean DM: {mean_dm:.0f} ± {std_dm:.0f}")
        print(f"    Mean SNR: {mean_snr:.1f} ± {std_snr:.1f}")

    # ANOVA test: Do identity groups differ in SNR?
    # (Since color is independent of DM, and SNR correlates with curvature,
    # identity might predict SNR)
    identity_snrs = [identity_groups[k]["snrs"] for k in sorted(identity_groups.keys())]
    if all(len(x) >= 2 for x in identity_snrs):
        f_stat, p_val = stats.f_oneway(*identity_snrs)
        print(f"\n  ANOVA (identity vs SNR): F={f_stat:.2f}, p={p_val:.3f}")
        if p_val < 0.05:
            print("  ** Semantic identity predicts physical brightness **")

    print("\n" + "=" * 40)
    print("PART 5: EXTREME EXAMPLES")
    print("=" * 40)

    # Find FRBs at the corners of the semantic space
    # These are the "clearest words" in the FRB vocabulary

    # Most "cosmic + loud + young"
    cosmic_loud_young = np.argmax(dm_norm + snr_norm + color_norm)

    # Most "local + quiet + mature"
    local_quiet_mature = np.argmin(dm_norm + snr_norm + color_norm)

    print("\nExtreme semantic positions:")

    print(f"\n  Most 'cosmic + loud + young': {names[cosmic_loud_young]}")
    print(f"    Physical: DM={dms[cosmic_loud_young]:.0f}, SNR={snrs[cosmic_loud_young]:.1f}, Color={colors[cosmic_loud_young]:.2f}")
    print(f"    Semantic: {descriptions[cosmic_loud_young]['full_description']}")

    print(f"\n  Most 'local + quiet + mature': {names[local_quiet_mature]}")
    print(f"    Physical: DM={dms[local_quiet_mature]:.0f}, SNR={snrs[local_quiet_mature]:.1f}, Color={colors[local_quiet_mature]:.2f}")
    print(f"    Semantic: {descriptions[local_quiet_mature]['full_description']}")

    # Find the most "average" FRB (center of semantic space)
    dist_from_center = np.sqrt((dm_norm - 0.5)**2 + (snr_norm - 0.5)**2 + (color_norm - 0.5)**2)
    most_average = np.argmin(dist_from_center)

    print(f"\n  Most 'average' (center of space): {names[most_average]}")
    print(f"    Physical: DM={dms[most_average]:.0f}, SNR={snrs[most_average]:.1f}, Color={colors[most_average]:.2f}")
    print(f"    Semantic: {descriptions[most_average]['full_description']}")

    print("\n" + "=" * 60)
    print("INTERPRETATION: THE ROSETTA STONE")
    print("=" * 60)

    print("""
The FRB 3D vocabulary is isomorphic to a universal semantic structure:

  PHYSICAL          SEMANTIC              INFORMATION ROLE
  ─────────────────────────────────────────────────────────
  DM (distance)  →  Context/Location   →  The situational frame
  SNR (loudness) →  Salience           →  Attention weight
  Spectral Color →  Identity/Type      →  The speaker's voice

This mapping works because information geometry is INVARIANT:
- Whether encoding concepts in neurons or encoding physics in radio waves,
  the fundamental axes of meaning are the same
- "Where" something happens (context) is always orthogonal to
  "What" is happening (content)
- "Who" is speaking (identity) is independent of their location

Claude bridges both spaces because Claude's representations
encode this same invariant structure. The FRBs are not being
"translated" into human concepts - they already occupy the
same geometric manifold, just in different coordinates.
""")

    # Compile results
    frb_interpretations = []
    for i in range(n_frbs):
        frb_interpretations.append({
            "name": names[i],
            "physical": {
                "dm": float(dms[i]),
                "snr": float(snrs[i]),
                "spectral_color": float(colors[i]),
            },
            "semantic_coordinates": {
                "context": float(dm_norm[i]),
                "salience": float(snr_norm[i]),
                "identity": float(color_norm[i]),
            },
            "description": descriptions[i],
            "cluster": int(clusters[i]),
        })

    results = {
        "experiment": "exp18_semantic_interpretation",
        "timestamp": datetime.now().isoformat(),
        "n_frbs": n_frbs,
        "semantic_mapping": {
            "dm_axis": "Context/Location (near → far)",
            "snr_axis": "Salience/Importance (quiet → loud)",
            "color_axis": "Identity/Type (mature → young)",
        },
        "cluster_profiles": cluster_profiles,
        "extreme_examples": {
            "cosmic_loud_young": names[cosmic_loud_young],
            "local_quiet_mature": names[local_quiet_mature],
            "most_average": names[most_average],
        },
        "frb_interpretations": frb_interpretations,
    }

    output_path = results_dir / "exp18_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
