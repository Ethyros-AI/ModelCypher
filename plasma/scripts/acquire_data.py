#!/usr/bin/env python3
"""
Data acquisition script for plasma research.

This script helps acquire tokamak diagnostic data from various sources.

Sources:
1. DisruptionBench - Standardized ML benchmark for disruption prediction
2. DIII-D public data - Via OSTI/DOE
3. FUSE simulations - Synthetic data from GA's FUSE.jl

Run from ModelCypher root:
    python plasma/scripts/acquire_data.py
"""

import os
import sys
from pathlib import Path


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_disruption_bench():
    """Check/download DisruptionBench data."""
    print_header("DisruptionBench")

    print("""
DisruptionBench is the primary target dataset for this research.

Paper: "Autoregressive Transformers for Disruption Prediction"
       arXiv:2401.00051

The dataset includes:
- Multi-device data (DIII-D, JET, others)
- Standardized format for ML
- Labeled disruption events
- 9 benchmark tasks

To acquire:
1. Check the paper's data availability statement
2. Contact authors (MIT PSFC):
   - Lucas Spangher
   - Cristina Rea (rea@psfc.mit.edu)
3. Or check: https://github.com/MIT-PSFC/disruption-predictor

If you have access, place data in:
    plasma/data/disruption_bench/
""")

    data_dir = Path(__file__).parent.parent / "data" / "disruption_bench"
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[OK] Data directory exists and has contents: {data_dir}")
    else:
        print(f"[  ] Data directory empty or missing: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)


def check_diiid_public():
    """Check DIII-D public datasets."""
    print_header("DIII-D Public Data")

    print("""
DIII-D public datasets are available through DOE/OSTI.

Primary sources:
1. OSTI Data Explorer
   https://www.osti.gov/dataexplorer/

   Search for: "DIII-D tokamak"

2. DIII-D Program Resources
   https://fusion.gat.com/global/diii-d/

3. Specific datasets:
   - "Initiation and Sustainment of Tokamak Plasmas"
     https://www.osti.gov/dataexplorer/biblio/dataset/1419641

Data access may require:
- Registration with DOE
- Institutional affiliation
- Agreeing to data use terms

If you have access, place data in:
    plasma/data/diiid/
""")

    data_dir = Path(__file__).parent.parent / "data" / "diiid"
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[OK] Data directory exists and has contents: {data_dir}")
    else:
        print(f"[  ] Data directory empty or missing: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)


def check_fuse_data():
    """Check FUSE simulation data."""
    print_header("FUSE Simulation Data")

    print("""
FUSE (Fusion Synthesis Engine) provides synthetic tokamak data.

GitHub: https://github.com/ProjectTorreyPines/FUSE.jl

Advantages:
- Open source (Apache 2.0)
- 200,000+ integrated simulations available
- Full control over parameters
- No access restrictions

Requirements:
- Julia 1.10+
- See FUSE.jl installation instructions

The FUSE Explorer dataset (announced 2026-01) includes:
- Comprehensive tokamak power plant designs
- Core plasma physics through to engineering

To generate synthetic data:

```julia
using FUSE
ini, act = FUSE.case_parameters(:FPP)
dd = FUSE.init(ini, act)
FUSE.ActorStationaryPlasma(dd, act)
# Export diagnostic data from dd
```

If you have FUSE output, place in:
    plasma/data/fuse/
""")

    data_dir = Path(__file__).parent.parent / "data" / "fuse"
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[OK] Data directory exists and has contents: {data_dir}")
    else:
        print(f"[  ] Data directory empty or missing: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)


def check_synthetic():
    """Verify synthetic data generation works."""
    print_header("Synthetic Data (for testing)")

    print("""
Synthetic plasma shots can be generated for pipeline testing.

This is NOT physically meaningful - just for code development.

Testing synthetic generation...
""")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from data_loader import create_synthetic_shot

        shot = create_synthetic_shot(disrupted=True, seed=42)
        print(f"[OK] Synthetic shot generated:")
        print(f"     - {shot.n_timesteps} timesteps")
        print(f"     - {shot.state_dim} dimensions")
        print(f"     - Disruption at t={shot.disruption_time:.2f}s")
    except Exception as e:
        print(f"[ERROR] Failed to generate synthetic data: {e}")


def main():
    print("""
============================================================
PLASMA DATA ACQUISITION
============================================================

This script checks data availability and provides instructions
for acquiring tokamak diagnostic data for the plasma geometry
research project.

The goal: Apply LLM geometry tools to plasma trajectories to
find geometric signatures that predict disruptions.
""")

    check_disruption_bench()
    check_diiid_public()
    check_fuse_data()
    check_synthetic()

    print_header("Summary")
    print("""
Priority order for data acquisition:

1. DisruptionBench (primary target)
   - Standardized format
   - Multiple devices
   - Existing ML baselines

2. FUSE synthetic data (for development)
   - No access barriers
   - Full parameter control
   - Quick iteration

3. DIII-D public data (if needed)
   - Real experimental data
   - May have access requirements

NEXT STEPS:
-----------
1. Contact MIT PSFC about DisruptionBench access
2. Install FUSE.jl for synthetic data generation
3. Run plasma/notebooks/01_data_exploration.py with synthetic data
""")


if __name__ == "__main__":
    main()
