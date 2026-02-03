#!/usr/bin/env python3
"""
Explore FAIR-MAST: Real high-dimensional tokamak diagnostic data.

FAIR-MAST provides ~30,000 shots from the MAST tokamak with full diagnostic data.
This is the high-dimensional data we need to test our geometry hypothesis.

S3 endpoint: https://s3.echo.stfc.ac.uk/mast/
Documentation: https://mastapp.site/

Run from ModelCypher root:
    python plasma/notebooks/03_fair_mast_exploration.py
"""

import sys
from pathlib import Path

import numpy as np

# Check dependencies
try:
    import xarray as xr
    import s3fs
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xarray", "s3fs", "zarr", "aiohttp"])
    import xarray as xr
    import s3fs


# FAIR-MAST S3 configuration
ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast"
LEVEL = "level1"


def get_s3_filesystem():
    """Get anonymous S3 filesystem for FAIR-MAST."""
    return s3fs.S3FileSystem(anon=True, endpoint_url=ENDPOINT_URL)


def list_available_shots(s3, limit: int = 20) -> list[int]:
    """List available shot numbers."""
    shots_path = f"{BUCKET}/{LEVEL}/shots/"
    items = s3.ls(shots_path)

    shot_ids = []
    for item in items[:limit]:
        # Extract shot number from path like "mast/level1/shots/30420.zarr"
        name = item.split("/")[-1]
        if name.endswith(".zarr"):
            shot_id = int(name.replace(".zarr", ""))
            shot_ids.append(shot_id)

    return sorted(shot_ids)


def list_diagnostics_for_shot(s3, shot_id: int) -> list[str]:
    """List diagnostic sources available for a shot."""
    shot_path = f"{BUCKET}/{LEVEL}/shots/{shot_id}.zarr/"
    try:
        items = s3.ls(shot_path)
        diagnostics = [item.split("/")[-1] for item in items if not item.endswith(".zattrs")]
        return diagnostics
    except Exception as e:
        print(f"Error listing diagnostics for shot {shot_id}: {e}")
        return []


def load_diagnostic(shot_id: int, diagnostic: str) -> xr.Dataset:
    """Load a diagnostic dataset for a shot."""
    url = f"{ENDPOINT_URL}/{BUCKET}/{LEVEL}/shots/{shot_id}.zarr/{diagnostic}"
    return xr.open_zarr(url)


def explore_diagnostic_structure(ds: xr.Dataset) -> dict:
    """Analyze the structure of a diagnostic dataset."""
    info = {
        "data_vars": list(ds.data_vars),
        "coords": list(ds.coords),
        "dims": dict(ds.dims),
        "attrs": dict(ds.attrs),
    }

    # Get shape info for each variable
    info["shapes"] = {name: ds[name].shape for name in ds.data_vars}
    info["dtypes"] = {name: str(ds[name].dtype) for name in ds.data_vars}

    return info


def main():
    print("=" * 70)
    print("FAIR-MAST EXPLORATION")
    print("Real high-dimensional tokamak diagnostic data")
    print("=" * 70)

    # Connect to S3
    print("\n1. Connecting to FAIR-MAST S3...")
    s3 = get_s3_filesystem()
    print(f"   Endpoint: {ENDPOINT_URL}")

    # List some shots
    print("\n2. Listing available shots...")
    try:
        shots = list_available_shots(s3, limit=10)
        print(f"   Found shots: {shots}")
    except Exception as e:
        print(f"   Error listing shots: {e}")
        print("   Trying direct access to known shot...")
        shots = [30420]  # Known good shot from documentation

    # Pick a shot and explore its diagnostics
    shot_id = shots[0] if shots else 30420
    print(f"\n3. Exploring shot {shot_id}...")

    diagnostics = list_diagnostics_for_shot(s3, shot_id)
    print(f"   Available diagnostics ({len(diagnostics)}):")
    for diag in diagnostics[:20]:
        print(f"     - {diag}")
    if len(diagnostics) > 20:
        print(f"     ... and {len(diagnostics) - 20} more")

    # Load and examine a few key diagnostics
    print("\n4. Loading sample diagnostics...")

    key_diagnostics = ["amc", "efm", "xim"]  # AMC=plasma current, EFM=EFIT, XIM=X-ray imaging

    total_channels = 0
    all_vars = []

    for diag in key_diagnostics:
        if diag in diagnostics:
            print(f"\n   Loading {diag}...")
            try:
                ds = load_diagnostic(shot_id, diag)
                info = explore_diagnostic_structure(ds)

                print(f"     Variables: {info['data_vars'][:5]}{'...' if len(info['data_vars']) > 5 else ''}")
                print(f"     Dims: {info['dims']}")

                # Count total channels
                for var, shape in info['shapes'].items():
                    if len(shape) > 0:
                        channels = np.prod(shape)
                        total_channels += channels
                        all_vars.append((diag, var, shape))

            except Exception as e:
                print(f"     Error: {e}")

    print(f"\n5. Data dimensionality summary:")
    print(f"   Total diagnostics available: {len(diagnostics)}")
    print(f"   Sample variables examined: {len(all_vars)}")
    print(f"   Total data points in sample: {total_channels:,}")

    # Try to load one diagnostic fully and check its structure
    print("\n6. Detailed look at AMC (magnetics) diagnostic...")
    try:
        ds = load_diagnostic(shot_id, "amc")
        print(f"   Dataset: {ds}")

        # Get time axis
        if 'time' in ds.coords:
            time = ds.coords['time'].values
            print(f"\n   Time range: {time[0]:.4f} to {time[-1]:.4f} seconds")
            print(f"   Time points: {len(time)}")

        # List all variables with shapes
        print(f"\n   Variables:")
        for var in list(ds.data_vars)[:15]:
            shape = ds[var].shape
            print(f"     {var}: {shape}")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
FAIR-MAST provides REAL high-dimensional tokamak data:
- ~30,000 shots available
- {len(diagnostics)} diagnostic sources per shot
- Time-resolved multi-channel measurements

This is the full-rank diagnostic data we need!

Next steps:
1. Download a batch of shots with disruption labels
2. Concatenate diagnostic channels into state vectors
3. Run geometry analysis on actual high-dimensional plasma trajectories
4. Test if disruption precursors have geometric signatures

The data is there. The tools are ready. Let's find the geometry.
""")


if __name__ == "__main__":
    main()
