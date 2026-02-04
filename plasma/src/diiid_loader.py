"""
Data loaders for DIII-D tokamak and ITPA Disruption Database.

Data Sources:
1. ITPA Disruption Database (Harvard Dataverse)
   - Multi-device database with scalar disruption parameters
   - Includes DIII-D, Alcator C-Mod, JET, ASDEX-U, MAST, NSTX
   - DOI: 10.7910/DVN/NXDX6U
   - Contains: equilibrium params, halo currents, current quench data

2. DisruptionPy (MIT-PSFC)
   - Python framework for MDSplus data retrieval
   - Requires credentials for direct DIII-D access
   - GitHub: https://github.com/MIT-PSFC/disruption-py

Limitations:
- ITPA database has scalar summary data, not full time-series
- Full time-series requires MDSplus access to DIII-D
- Cross-device comparison limited to shared scalar parameters
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json
import warnings

import numpy as np

from data_loader import PlasmaShot


@dataclass
class ITPADisruptionRecord:
    """Single disruption record from ITPA database.

    Contains scalar parameters from a disruption, not time-series.
    Useful for cross-device comparison of disruption characteristics.
    """
    shot_id: str
    device: str
    ip_max: float | None  # Peak plasma current [A]
    bt: float | None  # Toroidal field [T]
    q95: float | None  # Safety factor at 95% flux surface
    beta_n: float | None  # Normalized beta
    li: float | None  # Internal inductance
    ip_quench_rate: float | None  # Current quench rate [A/s]
    halo_fraction: float | None  # Halo current fraction
    radiated_fraction: float | None  # Radiated power fraction
    disruption_type: str | None  # Classification (VDE, density limit, etc.)
    metadata: dict


# Diagnostic mapping between devices
# Maps MAST diagnostic names to equivalent DIII-D names
MAST_TO_DIIID_MAPPING = {
    "plasma_current": "ip",
    "bt_vacuum": "bt",
    "q95": "q95",
    "beta_n": "betan",
    "li": "li",
    "elongation": "kappa",
    "triangularity_upper": "deltatriu",
    "triangularity_lower": "deltatril",
    "r_major": "rmajor",
    "a_minor": "aminor",
    "volume": "vol",
    "stored_energy": "wmhd",
    "n_e_line_avg": "nel",
}

# Shared scalar parameters available across devices in ITPA
SHARED_ITPA_PARAMS = [
    "ip_max",  # Peak plasma current
    "bt",  # Toroidal field
    "q95",  # Safety factor
    "beta_n",  # Normalized beta
    "li",  # Internal inductance
    "ip_quench_rate",  # CQ rate
]


class ITPALoader:
    """Load data from ITPA Disruption Database.

    The ITPA database contains scalar disruption parameters across multiple
    devices. It does NOT contain time-series data - only summary statistics
    for each disruption event.

    This is useful for:
    - Cross-device comparison of disruption severity
    - Statistical analysis of disruption types
    - Validating that different devices have similar disruption characteristics

    NOT useful for:
    - Time-series analysis
    - Training manifold models (no trajectories)
    - Early warning prediction (no temporal evolution)
    """

    def __init__(self, data_dir: Path | str | None = None):
        """Initialize ITPA loader.

        Args:
            data_dir: Path to downloaded ITPA data. If None, provides
                     instructions for downloading.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._records: list[ITPADisruptionRecord] | None = None

        if self.data_dir and not self.data_dir.exists():
            raise FileNotFoundError(self._download_instructions())

    def _download_instructions(self) -> str:
        return """
ITPA Disruption Database not found.

To download:
1. Visit https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NXDX6U
2. Download the dataset files
3. Extract to a local directory
4. Pass that directory to ITPALoader(data_dir=...)

Note: The ITPA database contains SCALAR parameters, not time-series.
For time-series disruption data, you need MDSplus access to individual
tokamaks (DIII-D, JET, etc.) via DisruptionPy.
"""

    def list_devices(self) -> list[str]:
        """List devices available in the database."""
        return ["DIII-D", "Alcator_C-Mod", "JET", "ASDEX_Upgrade", "MAST", "NSTX", "JT-60U", "TCV", "ADITYA"]

    def list_shots(self, device: str | None = None) -> list[str]:
        """List available shot IDs.

        Args:
            device: Filter by device name (optional)

        Returns:
            List of shot IDs
        """
        if self._records is None:
            self._load_records()

        shots = []
        for r in self._records:
            if device is None or r.device == device:
                shots.append(r.shot_id)
        return shots

    def load_record(self, shot_id: str) -> ITPADisruptionRecord:
        """Load a single disruption record.

        Args:
            shot_id: Shot identifier

        Returns:
            ITPADisruptionRecord with scalar parameters
        """
        if self._records is None:
            self._load_records()

        for r in self._records:
            if r.shot_id == shot_id:
                return r

        raise KeyError(f"Shot {shot_id} not found in ITPA database")

    def iter_records(
        self,
        device: str | None = None,
    ) -> Iterator[ITPADisruptionRecord]:
        """Iterate over disruption records.

        Args:
            device: Filter by device name (optional)

        Yields:
            ITPADisruptionRecord objects
        """
        if self._records is None:
            self._load_records()

        for r in self._records:
            if device is None or r.device == device:
                yield r

    def _load_records(self):
        """Load records from data files."""
        if self.data_dir is None:
            print(self._download_instructions())
            self._records = []
            return

        # TODO: Implement actual ITPA data parsing
        # The format depends on the specific files downloaded
        # This is a placeholder showing the expected structure
        warnings.warn(
            "ITPA data parsing not yet implemented. "
            "Download data and update _load_records() to parse the actual format."
        )
        self._records = []

    def get_shared_params_array(
        self,
        device: str | None = None,
        params: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Get shared parameters as array for cross-device comparison.

        Args:
            device: Filter by device (optional)
            params: Which parameters to include (default: SHARED_ITPA_PARAMS)

        Returns:
            data: [N, P] array of parameter values
            shot_ids: List of shot IDs
            param_names: List of parameter names
        """
        if params is None:
            params = SHARED_ITPA_PARAMS

        records = list(self.iter_records(device))
        if len(records) == 0:
            return np.array([]), [], params

        data = []
        shot_ids = []

        for r in records:
            row = []
            for p in params:
                val = getattr(r, p, None)
                row.append(val if val is not None else np.nan)
            data.append(row)
            shot_ids.append(r.shot_id)

        return np.array(data), shot_ids, params


class DisruptionPyLoader:
    """Interface to DisruptionPy for time-series data retrieval.

    DisruptionPy requires MDSplus server credentials for DIII-D access.
    This class provides a wrapper interface but requires external setup.

    Setup:
    1. pip install disruption-py
    2. Configure MDSplus credentials
    3. Request DIII-D data access from General Atomics

    GitHub: https://github.com/MIT-PSFC/disruption-py
    """

    def __init__(self, device: str = "d3d"):
        """Initialize DisruptionPy interface.

        Args:
            device: Device code ("d3d" for DIII-D, "cmod" for C-Mod)
        """
        self.device = device
        self._client = None

    def _check_setup(self):
        """Check if DisruptionPy is available and configured."""
        try:
            import disruption_py
            return True
        except ImportError:
            print("""
DisruptionPy not installed.

To install:
    pip install disruption-py

To use with DIII-D:
1. Contact General Atomics for data access
2. Configure MDSplus server credentials
3. See: https://github.com/MIT-PSFC/disruption-py
""")
            return False

    def load_shot(self, shot_id: int) -> PlasmaShot | None:
        """Load a shot via DisruptionPy (requires MDSplus access).

        Args:
            shot_id: DIII-D shot number

        Returns:
            PlasmaShot or None if access not available
        """
        if not self._check_setup():
            return None

        # TODO: Implement actual DisruptionPy integration
        # This requires MDSplus credentials and server access
        warnings.warn(
            "DisruptionPy integration not yet implemented. "
            "Requires MDSplus credentials for DIII-D access."
        )
        return None


def create_synthetic_diiid_shot(
    n_timesteps: int = 1000,
    disrupted: bool = False,
    seed: int | None = None,
) -> PlasmaShot:
    """Create synthetic DIII-D-like shot for testing cross-device analysis.

    This generates synthetic data with DIII-D-like diagnostic names and
    typical parameter ranges. For actual cross-device work, real data
    is needed.

    Args:
        n_timesteps: Number of time points
        disrupted: Whether to include disruption signature
        seed: Random seed

    Returns:
        PlasmaShot with DIII-D-like diagnostics
    """
    rng = np.random.default_rng(seed)

    # Time array (typical DIII-D shot ~5 seconds)
    time = np.linspace(0, 5.0, n_timesteps)

    # Generate DIII-D-like diagnostics
    # (typical parameter ranges for DIII-D)
    ip_base = 1e6 * (1.0 + 0.1 * rng.normal(0, 1))  # ~1 MA
    bt_base = 2.0 + 0.1 * rng.normal(0, 1)  # ~2 T

    # Create correlated diagnostic evolution
    n_diag = 30
    state = np.zeros((n_timesteps, n_diag))
    state[0] = rng.normal(0, 1, n_diag)

    for t in range(1, n_timesteps):
        state[t] = 0.99 * state[t-1] + 0.1 * rng.normal(0, 1, n_diag)

    # Add disruption signature if requested
    disruption_time = None
    if disrupted:
        disruption_time = 4.0 + rng.uniform(-0.5, 0.5)
        disrupt_idx = int(disruption_time / 5.0 * n_timesteps)

        for t in range(max(0, disrupt_idx - 100), disrupt_idx):
            progress = (t - (disrupt_idx - 100)) / 100
            state[t] += progress * rng.normal(0, 2, n_diag)

        for t in range(disrupt_idx, n_timesteps):
            state[t] = rng.normal(0, 0.1, n_diag)

    # Package with DIII-D-like diagnostic names
    diagnostics = {
        "ip": state[:, 0:1] * 1e6 + ip_base,  # Plasma current [A]
        "bt": state[:, 1:2] * 0.1 + bt_base,  # Toroidal field [T]
        "q95": state[:, 2:3] * 0.5 + 4.0,  # Safety factor
        "betan": state[:, 3:4] * 0.5 + 2.0,  # Normalized beta
        "li": state[:, 4:5] * 0.1 + 0.8,  # Internal inductance
        "nel": state[:, 5:6] * 0.5e19 + 3e19,  # Line-avg density [m^-3]
        "wmhd": state[:, 6:7] * 0.1e6 + 0.5e6,  # Stored energy [J]
        "pnbi": state[:, 7:8] * 1e6 + 5e6,  # NBI power [W]
        "drsep": state[:, 8:9] * 0.01,  # Separatrix position [m]
        "magnetic_probes": state[:, 9:25],  # 16 Mirnov coils
        "coil_currents": state[:, 25:30],  # 5 PF coils
    }

    return PlasmaShot(
        shot_id=f"synthetic_diiid_{seed}",
        device="DIII-D",
        time=time,
        diagnostics=diagnostics,
        disrupted=disrupted,
        disruption_time=disruption_time,
        metadata={"synthetic": True, "seed": seed},
    )


if __name__ == "__main__":
    print("DIII-D / ITPA Data Loader Test")
    print("=" * 50)

    # Test synthetic DIII-D shot
    print("\n1. Creating synthetic DIII-D shot...")
    shot = create_synthetic_diiid_shot(disrupted=True, seed=42)
    print(f"   Shot ID: {shot.shot_id}")
    print(f"   Device: {shot.device}")
    print(f"   Timesteps: {shot.n_timesteps}")
    print(f"   State dim: {shot.state_dim}")
    print(f"   Diagnostics: {list(shot.diagnostics.keys())}")
    print(f"   Disrupted: {shot.disrupted} at t={shot.disruption_time:.2f}s" if shot.disrupted else "   Stable")

    # Show ITPA loader status
    print("\n2. ITPA Disruption Database:")
    loader = ITPALoader(data_dir=None)  # Will print download instructions
    print(f"   Available devices: {loader.list_devices()}")

    # Show diagnostic mapping
    print("\n3. MAST ↔ DIII-D diagnostic mapping:")
    for mast, diiid in list(MAST_TO_DIIID_MAPPING.items())[:5]:
        print(f"   {mast} → {diiid}")
    print("   ...")
