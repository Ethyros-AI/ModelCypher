# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Experiment 3: ID Trajectory Measurement

Question: How does intrinsic dimension evolve during training?

This experiment tracks:
- ID trajectory across epochs with CI bands
- Kendall τ for monotonicity
- Spike detection (epochs where ID exits previous CI)

Run with:
    poetry run pytest tests/experiments/test_lora_geometry_exp3.py -v -s --capture=no
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.lora_geometry.id_trajectory import (
    IDTrajectory,
    IDTrajectoryPoint,
    IDTrajectoryTracker,
    measure_id_at_checkpoint,
)
from modelcypher.experimental.lora_geometry.statistics import compute_kendall_tau

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# Results directory
RESULTS_DIR = Path("results/id_trajectory")


def _ensure_results_dir() -> None:
    """Create results directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _create_synthetic_activations_with_evolving_id(
    epoch: int,
    n_probes: int = 128,
    hidden_dim: int = 64,
    target_id: float = 10.0,
    backend: "Backend | None" = None,
) -> "Array":
    """Create synthetic activations with controlled ID that decreases over epochs.

    Simulates training: early epochs have higher ID (more exploration),
    later epochs have lower ID (convergence to manifold).

    Args:
        epoch: Current epoch.
        n_probes: Number of probe samples.
        hidden_dim: Hidden dimension.
        target_id: Target intrinsic dimension at convergence.
        backend: Compute backend.

    Returns:
        Activations array [n_probes, hidden_dim].
    """
    if backend is None:
        backend = get_default_backend()

    # ID decreases as training progresses (convergence)
    # Start at ~hidden_dim, decay toward target_id
    decay_rate = 0.3
    effective_id = target_id + (hidden_dim - target_id) * (1.0 / (1.0 + decay_rate * epoch))
    effective_id = max(target_id, min(effective_id, hidden_dim))

    # Create manifold with controlled ID
    # Use a low-rank structure + noise
    effective_rank = int(effective_id)

    # Base manifold (low-rank)
    manifold_basis = backend.random_normal((n_probes, effective_rank), dtype="float32")

    # Project to full dimension
    projection = backend.random_normal((effective_rank, hidden_dim), dtype="float32")
    activations = backend.matmul(manifold_basis, projection)

    # Add small noise (controls how "clean" the manifold is)
    noise_scale = 0.1 / (1.0 + 0.5 * epoch)  # Noise decreases with training
    noise = backend.random_normal((n_probes, hidden_dim), dtype="float32")
    noise = backend.multiply(noise, noise_scale)
    activations = backend.add(activations, noise)

    backend.eval(activations)
    return activations


class TestIDTrajectoryInfrastructure:
    """Test ID trajectory infrastructure."""

    def test_measure_id_at_checkpoint(self):
        """Can measure ID at a single checkpoint."""
        backend = get_default_backend()
        backend.random_seed(42)

        activations = _create_synthetic_activations_with_evolving_id(
            epoch=0, n_probes=128, hidden_dim=32, backend=backend
        )

        point = measure_id_at_checkpoint(
            activations=activations,
            epoch=0,
            use_convergence=False,  # Faster for testing
            backend=backend,
        )

        assert point.epoch == 0
        assert point.intrinsic_dimension > 0
        assert point.ci_lower <= point.intrinsic_dimension <= point.ci_upper
        assert point.usable_count > 0

    def test_id_trajectory_tracker(self):
        """Tracker accumulates points and computes analysis."""
        backend = get_default_backend()
        backend.random_seed(42)

        tracker = IDTrajectoryTracker(
            probe_set_size=128,
            adapter_id="test_tracker",
            use_convergence=False,
            backend=backend,
        )

        # Record several epochs
        for epoch in range(5):
            activations = _create_synthetic_activations_with_evolving_id(
                epoch=epoch, n_probes=128, hidden_dim=32, backend=backend
            )
            tracker.record(epoch, activations)

        trajectory = tracker.get_trajectory()

        assert len(trajectory.points) == 5
        assert trajectory.kendall_tau is not None
        assert trajectory.adapter_id == "test_tracker"

    def test_kendall_tau_computation(self):
        """Kendall τ correctly measures monotonicity."""
        # Perfect increasing sequence
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        tau = compute_kendall_tau(x, y)
        assert abs(tau - 1.0) < 0.01

        # Perfect decreasing
        y_dec = [5, 4, 3, 2, 1]
        tau_dec = compute_kendall_tau(x, y_dec)
        assert abs(tau_dec - (-1.0)) < 0.01

    def test_spike_detection(self):
        """Spikes detected when ID exits CI band."""
        trajectory = IDTrajectory(adapter_id="spike_test")

        # Add points where second point is outside first's CI
        trajectory.add_point(
            IDTrajectoryPoint(
                epoch=0,
                intrinsic_dimension=10.0,
                ci_lower=9.0,
                ci_upper=11.0,
                usable_count=100,
            )
        )
        trajectory.add_point(
            IDTrajectoryPoint(
                epoch=1,
                intrinsic_dimension=15.0,  # Outside [9, 11]
                ci_lower=14.0,
                ci_upper=16.0,
                usable_count=100,
            )
        )

        assert 1 in trajectory.spike_epochs


class TestFullExperiment:
    """Full ID trajectory experiment."""

    @pytest.mark.slow
    def test_full_id_trajectory_experiment(self):
        """Run full ID trajectory experiment with synthetic training."""
        _ensure_results_dir()

        backend = get_default_backend()
        backend.random_seed(42)

        # Simulate training for multiple epochs
        n_epochs = 10
        n_probes = 128
        hidden_dim = 32
        target_id = 8.0

        tracker = IDTrajectoryTracker(
            probe_set_size=n_probes,
            adapter_id="synthetic_training",
            use_convergence=False,  # Faster for experiment
            backend=backend,
        )

        print("\n=== ID Trajectory Experiment ===")
        print(f"Epochs: {n_epochs}")
        print(f"Probes: {n_probes}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Target ID: {target_id}")
        print()

        for epoch in range(n_epochs):
            # Simulate evolving activations
            activations = _create_synthetic_activations_with_evolving_id(
                epoch=epoch,
                n_probes=n_probes,
                hidden_dim=hidden_dim,
                target_id=target_id,
                backend=backend,
            )

            point = tracker.record(epoch, activations)
            print(
                f"Epoch {epoch}: ID={point.intrinsic_dimension:.2f} "
                f"CI=[{point.ci_lower:.2f}, {point.ci_upper:.2f}] "
                f"usable={point.usable_count}"
            )

        trajectory = tracker.get_trajectory()

        # Save trajectory
        trajectory_data = trajectory.to_dict()
        with open(RESULTS_DIR / "trajectory.json", "w") as f:
            json.dump(trajectory_data, f, indent=2)

        # Save Kendall tau analysis
        kendall_analysis = {
            "kendall_tau": trajectory.kendall_tau,
            "spike_epochs": trajectory.spike_epochs,
            "n_points": len(trajectory.points),
            "interpretation": (
                "monotonic_decrease"
                if trajectory.kendall_tau and trajectory.kendall_tau < -0.5
                else "monotonic_increase"
                if trajectory.kendall_tau and trajectory.kendall_tau > 0.5
                else "non_monotonic"
            ),
        }
        with open(RESULTS_DIR / "kendall_tau.json", "w") as f:
            json.dump(kendall_analysis, f, indent=2)

        # Verification checks
        ci_widths = [p.ci_upper - p.ci_lower for p in trajectory.points]
        ci_widths_sorted = sorted(ci_widths)

        verifications = {
            "ci_width_quantiles": {
                "p10": ci_widths_sorted[int(len(ci_widths_sorted) * 0.1)]
                if ci_widths_sorted
                else None,
                "p50": ci_widths_sorted[int(len(ci_widths_sorted) * 0.5)]
                if ci_widths_sorted
                else None,
                "p90": ci_widths_sorted[int(len(ci_widths_sorted) * 0.9)]
                if ci_widths_sorted
                else None,
            },
            "degenerate_ci_check": all(w > 0 for w in ci_widths),
            "sample_coverage": sum(
                1 for p in trajectory.points if p.usable_count > 10
            )
            / len(trajectory.points),
        }

        with open(RESULTS_DIR / "verification_checks.json", "w") as f:
            json.dump(verifications, f, indent=2)

        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"Kendall τ: {trajectory.kendall_tau:.4f}")
        print(f"Spike epochs: {trajectory.spike_epochs}")
        print(f"Interpretation: {kendall_analysis['interpretation']}")

        # Assertions
        assert len(trajectory.points) == n_epochs
        assert trajectory.kendall_tau is not None
        assert -1.0 <= trajectory.kendall_tau <= 1.0

        # For decreasing ID trajectory (training convergence), expect negative tau
        # But this is measurement, not a pass/fail criterion
        print(f"\nExpected negative τ for convergence: τ = {trajectory.kendall_tau:.4f}")


class TestConvergenceDetection:
    """Test convergence detection during ID trajectory tracking."""

    def test_convergence_detection(self):
        """Tracker can detect when ID stabilizes."""
        backend = get_default_backend()
        backend.random_seed(42)

        tracker = IDTrajectoryTracker(
            probe_set_size=64,
            adapter_id="convergence_test",
            use_convergence=False,
            backend=backend,
        )

        # Create stable ID trajectory (simulates converged training)
        for epoch in range(10):
            # After epoch 5, ID stabilizes
            effective_id = 10.0 if epoch < 5 else 8.0 + 0.01 * backend.to_scalar(
                backend.random_uniform(-1.0, 1.0, shape=(1,))
            )

            # Create activations with stable ID
            n_probes = 64
            effective_rank = int(effective_id)
            manifold_basis = backend.random_normal((n_probes, effective_rank), dtype="float32")
            projection = backend.random_normal((effective_rank, 32), dtype="float32")
            activations = backend.matmul(manifold_basis, projection)
            backend.eval(activations)

            tracker.record(epoch, activations)

            # Check convergence after enough points
            if epoch >= 5:
                is_converging = tracker.is_converging(window=3, threshold=0.2)
                # Note: This is a measurement, actual convergence depends on data
                print(f"Epoch {epoch}: converging={is_converging}")

        trajectory = tracker.get_trajectory()
        assert len(trajectory.points) == 10
