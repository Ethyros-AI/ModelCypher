"""
Data loaders for tokamak diagnostic data.

Supports:
- DisruptionBench format (primary)
- DIII-D public datasets
- FUSE simulation output
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json
import warnings

import numpy as np

# Suppress benign numpy warnings from random number generation
warnings.filterwarnings("ignore", category=RuntimeWarning, module="data_loader")


@dataclass
class PlasmaShot:
    """A single tokamak shot (discharge).

    Attributes:
        shot_id: Unique identifier for this shot
        device: Tokamak name (e.g., "DIII-D", "JET", "KSTAR")
        time: Array of time points (seconds)
        diagnostics: Dict mapping diagnostic name to [time, channels] array
        disrupted: Whether this shot ended in a disruption
        disruption_time: Time of disruption (if applicable)
        metadata: Additional shot information
    """
    shot_id: str
    device: str
    time: np.ndarray  # [T]
    diagnostics: dict[str, np.ndarray]  # name -> [T, C] array
    disrupted: bool
    disruption_time: float | None
    metadata: dict

    @property
    def n_timesteps(self) -> int:
        return len(self.time)

    @property
    def state_dim(self) -> int:
        """Total dimension of diagnostic state vector."""
        return sum(d.shape[1] for d in self.diagnostics.values())

    def get_state_vector(self, t_idx: int) -> np.ndarray:
        """Get concatenated diagnostic state at time index t_idx."""
        return np.concatenate([
            self.diagnostics[name][t_idx]
            for name in sorted(self.diagnostics.keys())
        ])

    def get_trajectory(self) -> np.ndarray:
        """Get full trajectory as [T, D] array."""
        return np.stack([
            self.get_state_vector(t) for t in range(self.n_timesteps)
        ])


class DisruptionBenchLoader:
    """Load data from DisruptionBench format.

    DisruptionBench provides standardized tokamak data for ML research.
    Paper: https://arxiv.org/abs/2401.00051
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"DisruptionBench data not found at {data_dir}. "
                "Download from: https://github.com/MIT-PSFC/disruption-bench"
            )

    def list_shots(self, device: str | None = None) -> list[str]:
        """List available shot IDs, optionally filtered by device."""
        # TODO: Implement based on actual DisruptionBench structure
        raise NotImplementedError("Waiting for DisruptionBench data")

    def load_shot(self, shot_id: str) -> PlasmaShot:
        """Load a single shot by ID."""
        # TODO: Implement based on actual DisruptionBench structure
        raise NotImplementedError("Waiting for DisruptionBench data")

    def iter_shots(
        self,
        device: str | None = None,
        disrupted_only: bool = False,
        stable_only: bool = False,
    ) -> Iterator[PlasmaShot]:
        """Iterate over shots matching criteria."""
        for shot_id in self.list_shots(device):
            shot = self.load_shot(shot_id)
            if disrupted_only and not shot.disrupted:
                continue
            if stable_only and shot.disrupted:
                continue
            yield shot


class FUSELoader:
    """Load data from FUSE.jl simulation output.

    FUSE provides synthetic tokamak data following ITER IMAS ontology.
    GitHub: https://github.com/ProjectTorreyPines/FUSE.jl
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_shot(self, shot_id: str) -> PlasmaShot:
        """Load a FUSE simulation shot."""
        # TODO: Implement based on FUSE output format
        raise NotImplementedError("Waiting for FUSE data")


def create_synthetic_shot(
    n_timesteps: int = 1000,
    n_diagnostics: int = 50,
    disrupted: bool = False,
    seed: int | None = None,
) -> PlasmaShot:
    """Create synthetic plasma shot for testing.

    Generates plausible diagnostic trajectories with:
    - Smooth temporal evolution
    - Correlated diagnostic channels
    - Optional disruption signature

    This is for pipeline testing only - not physically meaningful.
    """
    rng = np.random.default_rng(seed)

    # Time array (0 to 5 seconds, typical shot duration)
    time = np.linspace(0, 5.0, n_timesteps)

    # Generate correlated random walk for base state
    # This creates temporally smooth, spatially correlated structure
    state = np.zeros((n_timesteps, n_diagnostics), dtype=np.float64)
    state[0] = rng.normal(0, 1, n_diagnostics)

    # Correlation matrix for spatial structure
    distances = np.abs(np.arange(n_diagnostics)[:, None] - np.arange(n_diagnostics))
    correlation = np.exp(-distances / 10).astype(np.float64)  # Exponential decay
    L = np.linalg.cholesky(correlation + 0.01 * np.eye(n_diagnostics, dtype=np.float64))

    for t in range(1, n_timesteps):
        noise = L @ rng.normal(0, 0.1, n_diagnostics).astype(np.float64)
        state[t] = 0.99 * state[t-1] + noise

    # Add disruption signature if requested
    disruption_time = None
    if disrupted:
        disruption_time = 4.0 + rng.uniform(-0.5, 0.5)
        disruption_idx = int(disruption_time / 5.0 * n_timesteps)

        # Pre-disruption: growing instability
        for t in range(max(0, disruption_idx - 100), disruption_idx):
            progress = (t - (disruption_idx - 100)) / 100
            state[t] += progress * rng.normal(0, 2, n_diagnostics)

        # Post-disruption: collapse
        for t in range(disruption_idx, n_timesteps):
            state[t] = rng.normal(0, 0.1, n_diagnostics)

    # Package into diagnostic dict
    diagnostics = {
        "electron_temp": state[:, :10],
        "electron_density": state[:, 10:20],
        "ion_temp": state[:, 20:30],
        "magnetic": state[:, 30:40],
        "radiation": state[:, 40:50],
    }

    return PlasmaShot(
        shot_id=f"synthetic_{seed}",
        device="synthetic",
        time=time,
        diagnostics=diagnostics,
        disrupted=disrupted,
        disruption_time=disruption_time,
        metadata={"synthetic": True, "seed": seed},
    )


if __name__ == "__main__":
    # Test synthetic data generation
    print("Creating synthetic plasma shots...")

    stable = create_synthetic_shot(disrupted=False, seed=42)
    print(f"Stable shot: {stable.n_timesteps} timesteps, {stable.state_dim} dimensions")

    disrupted = create_synthetic_shot(disrupted=True, seed=43)
    print(f"Disrupted shot: disruption at t={disrupted.disruption_time:.2f}s")

    # Get trajectory
    traj = stable.get_trajectory()
    print(f"Trajectory shape: {traj.shape}")
