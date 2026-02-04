"""
Physical interpretation of PCA manifold components for plasma diagnostics.

Maps PCA loadings to MAST diagnostic names and physics categories.
Provides human-readable interpretation of manifold gradients.
"""

import numpy as np
from dataclasses import dataclass

from geometry_tools import PCAManifold, ManifoldGradient


# MAST diagnostic categories based on AMC (Acquisition and Machine Control) system
# Reference: FAIR-MAST data documentation
MAST_PHYSICS_CATEGORIES = {
    "plasma_current": {
        "description": "Total toroidal plasma current (Ip)",
        "unit": "A",
        "physics": "Global stability indicator",
    },
    "coil_currents": {
        "description": "Poloidal field coil currents",
        "signals": ["p2_current", "p3_current", "p4_current", "p5_current", "p6_current"],
        "unit": "A",
        "physics": "Vertical/radial position control",
    },
    "solenoid": {
        "description": "Central solenoid currents for flux swing",
        "signals": ["solenoid_current", "solenoid_voltage"],
        "unit": "A, V",
        "physics": "Inductive current drive",
    },
    "magnetic_probes": {
        "description": "Local magnetic field measurements",
        "signals": ["bp_*", "br_*", "bz_*"],  # Wildcard patterns
        "unit": "T",
        "physics": "MHD mode detection, equilibrium reconstruction",
    },
    "rogowski_coils": {
        "description": "Current measurement coils",
        "unit": "A",
        "physics": "Integrated current measurement",
    },
    "flux_loops": {
        "description": "Poloidal flux measurements",
        "unit": "Wb",
        "physics": "Equilibrium reconstruction",
    },
    "position_control": {
        "description": "Plasma position estimates and targets",
        "signals": ["r_target", "z_target", "r_actual", "z_actual"],
        "unit": "m",
        "physics": "Horizontal/vertical position",
    },
    "shape_control": {
        "description": "Plasma shape parameters",
        "signals": ["elongation", "triangularity", "minor_radius"],
        "unit": "dimensionless, m",
        "physics": "MHD stability (shaping)",
    },
}


@dataclass
class DiagnosticInfo:
    """Information about a diagnostic channel."""
    name: str
    index: int
    category: str | None
    description: str | None
    unit: str | None


def classify_diagnostic(name: str) -> tuple[str | None, str | None]:
    """Classify a diagnostic name into physics category.

    Args:
        name: Diagnostic signal name (e.g., "p3_current", "bp_upper_03")

    Returns:
        (category, description) or (None, None) if unrecognized
    """
    name_lower = name.lower()

    # Check for exact or pattern matches
    if "plasma_current" in name_lower or name_lower == "ip":
        return "plasma_current", "Total plasma current"

    if any(coil in name_lower for coil in ["p2_", "p3_", "p4_", "p5_", "p6_", "pf_"]):
        return "coil_currents", f"PF coil: {name}"

    if "solenoid" in name_lower or name_lower.startswith("cs"):
        return "solenoid", "Central solenoid"

    if any(prefix in name_lower for prefix in ["bp_", "br_", "bz_", "b_"]):
        return "magnetic_probes", f"Magnetic probe: {name}"

    if "rogowski" in name_lower:
        return "rogowski_coils", "Rogowski coil current"

    if "flux" in name_lower or "psi" in name_lower:
        return "flux_loops", "Poloidal flux measurement"

    if any(pos in name_lower for pos in ["_target", "_actual", "r_", "z_", "position"]):
        return "position_control", "Position control signal"

    if any(shape in name_lower for shape in ["elongation", "kappa", "delta", "triangularity"]):
        return "shape_control", "Shape parameter"

    return None, None


def build_diagnostic_map(names: list[str]) -> dict[int, DiagnosticInfo]:
    """Build mapping from feature indices to diagnostic information.

    Args:
        names: List of diagnostic names (in same order as feature vector)

    Returns:
        Dict mapping index to DiagnosticInfo
    """
    diag_map = {}
    for idx, name in enumerate(names):
        category, description = classify_diagnostic(name)
        diag_map[idx] = DiagnosticInfo(
            name=name,
            index=idx,
            category=category,
            description=description,
            unit=MAST_PHYSICS_CATEGORIES.get(category, {}).get("unit"),
        )
    return diag_map


def interpret_pc_loadings(
    pca_model: PCAManifold,
    pc_idx: int = 0,
    n_top: int = 10,
) -> dict:
    """Interpret what a principal component represents physically.

    Args:
        pca_model: Fitted PCA manifold with diagnostic_names
        pc_idx: Which PC to interpret (0-indexed)
        n_top: Number of top contributors to report

    Returns:
        Dict with physical interpretation including:
        - variance_explained: Fraction of variance this PC captures
        - top_contributors: List of (name, loading, category)
        - category_summary: Aggregated contribution by physics category
        - interpretation: Human-readable description
    """
    if pca_model.diagnostic_names is None:
        raise ValueError("PCAManifold must have diagnostic_names for interpretation")

    loadings = pca_model.components[pc_idx]
    var_ratio = pca_model.explained_variance_ratio[pc_idx]

    # Get top contributors by absolute loading
    sorted_idx = np.argsort(np.abs(loadings))[::-1][:n_top]

    top_contributors = []
    category_totals = {}

    for idx in sorted_idx:
        name = pca_model.diagnostic_names[idx]
        loading = loadings[idx]
        category, desc = classify_diagnostic(name)

        top_contributors.append({
            "name": name,
            "index": idx,
            "loading": float(loading),
            "abs_loading": float(np.abs(loading)),
            "category": category,
            "description": desc,
        })

    # Aggregate by category (sum of squared loadings)
    for idx, loading in enumerate(loadings):
        name = pca_model.diagnostic_names[idx]
        category, _ = classify_diagnostic(name)
        if category is None:
            category = "unknown"
        category_totals[category] = category_totals.get(category, 0) + loading**2

    # Normalize category totals
    total_loading = sum(category_totals.values())
    category_summary = {
        cat: {
            "contribution": float(val / total_loading) if total_loading > 0 else 0,
            "description": MAST_PHYSICS_CATEGORIES.get(cat, {}).get("description", "Unknown"),
            "physics": MAST_PHYSICS_CATEGORIES.get(cat, {}).get("physics", "Unknown"),
        }
        for cat, val in sorted(category_totals.items(), key=lambda x: -x[1])
    }

    # Generate interpretation
    top_category = max(category_totals.items(), key=lambda x: x[1])[0]
    interpretation = _generate_pc_interpretation(
        pc_idx, var_ratio, top_contributors, top_category
    )

    return {
        "pc_index": pc_idx,
        "variance_explained": float(var_ratio),
        "top_contributors": top_contributors,
        "category_summary": category_summary,
        "interpretation": interpretation,
    }


def _generate_pc_interpretation(
    pc_idx: int,
    var_ratio: float,
    top_contributors: list[dict],
    top_category: str,
) -> str:
    """Generate human-readable interpretation of a PC."""
    lines = [f"PC{pc_idx+1} ({var_ratio*100:.1f}% variance):"]

    # Dominant category
    cat_info = MAST_PHYSICS_CATEGORIES.get(top_category, {})
    if cat_info:
        lines.append(f"  Dominated by: {cat_info.get('description', top_category)}")
        lines.append(f"  Physical meaning: {cat_info.get('physics', 'Unknown')}")

    # Top signals
    lines.append("  Key signals:")
    for contrib in top_contributors[:5]:
        sign = "+" if contrib["loading"] > 0 else "-"
        lines.append(f"    {sign} {contrib['name']}: {contrib['abs_loading']:.3f}")

    return "\n".join(lines)


def interpret_gradient(
    gradient: ManifoldGradient,
    pca_model: PCAManifold,
) -> dict:
    """Interpret what a manifold gradient means physically.

    Args:
        gradient: ManifoldGradient from compute_gradient_to_manifold
        pca_model: PCA model used to compute the gradient

    Returns:
        Dict with physical interpretation including:
        - drift_direction: Which physics categories are driving the drift
        - correction_needed: What changes would return to stability
        - severity: How far from stable manifold
    """
    if pca_model.diagnostic_names is None:
        return {
            "error": "No diagnostic names available for interpretation",
            "distance": gradient.distance,
        }

    # Analyze which features drive the deviation
    feature_deviations = {}
    category_deviations = {}

    for idx, deviation in enumerate(gradient.gradient_by_feature):
        name = pca_model.diagnostic_names[idx]
        category, _ = classify_diagnostic(name)
        if category is None:
            category = "unknown"

        feature_deviations[name] = float(deviation)
        category_deviations[category] = category_deviations.get(category, 0) + deviation**2

    # Normalize by total deviation
    total_dev = sum(category_deviations.values())
    if total_dev > 0:
        category_contributions = {
            cat: float(dev / total_dev)
            for cat, dev in category_deviations.items()
        }
    else:
        category_contributions = {}

    # Severity assessment
    if gradient.distance < 0.5:
        severity = "low"
        severity_desc = "State is near the stable manifold"
    elif gradient.distance < 2.0:
        severity = "moderate"
        severity_desc = "Notable deviation from normal operation"
    elif gradient.distance < 5.0:
        severity = "high"
        severity_desc = "Significant deviation - potential disruption precursor"
    else:
        severity = "critical"
        severity_desc = "Far from stable manifold - disruption likely imminent"

    # Correction summary
    corrections = []
    for feat_idx, deviation, name in gradient.top_driving_features[:5]:
        if name:
            direction = "increase" if deviation > 0 else "decrease"
            corrections.append(f"{direction} {name}")

    return {
        "distance": gradient.distance,
        "reconstruction_error": gradient.reconstruction_error,
        "severity": severity,
        "severity_description": severity_desc,
        "category_contributions": category_contributions,
        "top_deviations": {
            pca_model.diagnostic_names[idx] if pca_model.diagnostic_names else f"feature_{idx}": float(dev)
            for idx, dev, _ in gradient.top_driving_features[:5]
        },
        "correction_needed": corrections,
    }


def summarize_manifold(pca_model: PCAManifold, n_pcs: int = 5) -> str:
    """Generate human-readable summary of what the manifold captures.

    Args:
        pca_model: Fitted PCA manifold with diagnostic names
        n_pcs: Number of top PCs to summarize

    Returns:
        Multi-line string summary
    """
    lines = [
        "=" * 60,
        "PLASMA STABILITY MANIFOLD SUMMARY",
        "=" * 60,
        f"Total diagnostics: {pca_model.n_features}",
        f"Manifold dimension: {pca_model.n_components}",
        f"Total variance captured: {pca_model.explained_variance_ratio.sum()*100:.1f}%",
        "",
    ]

    if pca_model.diagnostic_names:
        for i in range(min(n_pcs, pca_model.n_components)):
            interp = interpret_pc_loadings(pca_model, i, n_top=5)
            lines.append(interp["interpretation"])
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with synthetic data that has named diagnostics
    from geometry_tools import compute_pca_manifold
    from data_loader import create_synthetic_shot

    print("Testing manifold interpreter...")

    # Create synthetic shots with named diagnostics
    shots = []
    for i in range(5):
        shot = create_synthetic_shot(disrupted=False, seed=i)
        shots.append(shot.get_trajectory())

    # Create diagnostic names matching synthetic data structure
    diag_names = []
    for prefix in ["electron_temp", "electron_density", "ion_temp", "magnetic", "radiation"]:
        for j in range(10):
            diag_names.append(f"{prefix}_{j:02d}")

    # Fit manifold
    manifold = compute_pca_manifold(shots, n_components=5, diagnostic_names=diag_names)

    # Interpret
    print(summarize_manifold(manifold))

    # Test individual PC interpretation
    print("\nDetailed PC1 interpretation:")
    pc1_interp = interpret_pc_loadings(manifold, 0, n_top=10)
    for cat, info in pc1_interp["category_summary"].items():
        print(f"  {cat}: {info['contribution']*100:.1f}%")
