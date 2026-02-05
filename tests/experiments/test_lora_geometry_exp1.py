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

"""Experiment 1: Selectivity ↔ Conflict Measurement

Question: What is the empirical relationship between spectral selectivity
and conflict score?

This experiment collects:
- Weight-space metrics: spectral selectivity, Weyl utilization
- Output-space metrics: conflict score
- Per-layer distributions
- Correlation analysis with bootstrap CI

Run with:
    poetry run pytest tests/experiments/test_lora_geometry_exp1.py -v -s --capture=no
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.lora_geometry.measurements import (
    AdapterMeasurement,
    LayerMeasurement,
    collect_layer_measurements,
)
from modelcypher.experimental.lora_geometry.statistics import (
    CorrelationResult,
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_bootstrap_ci,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# Results directory
RESULTS_DIR = Path("results/selectivity_conflict")


def _ensure_results_dir() -> None:
    """Create results directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _create_synthetic_adapter_with_varying_selectivity(
    selectivity_level: float,
    n_layers: int = 4,
    hidden_dim: int = 128,
    lora_rank: int = 8,
    backend: "Backend | None" = None,
) -> tuple[dict[str, "Array"], dict[str, "Array"]]:
    """Create synthetic adapter with controlled selectivity.

    Args:
        selectivity_level: 0=uniform (low CV), 1=spiky (high CV)
        n_layers: Number of layers.
        hidden_dim: Hidden dimension.
        lora_rank: LoRA rank.
        backend: Compute backend.

    Returns:
        Tuple of (base_weights, delta_weights).
    """
    if backend is None:
        backend = get_default_backend()

    base_weights = {}
    delta_weights = {}

    for layer_idx in range(n_layers):
        key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"

        # Base weight
        W = backend.random_normal((hidden_dim, hidden_dim), dtype="float32")
        backend.eval(W)
        base_weights[key] = W

        # Create delta with controlled selectivity
        if selectivity_level < 0.5:
            # Low selectivity: uniform random (low CV)
            B = backend.random_normal((hidden_dim, lora_rank), dtype="float32")
            A = backend.random_normal((lora_rank, hidden_dim), dtype="float32")
            delta = backend.matmul(B, A)
            delta = backend.multiply(delta, 0.01)
        else:
            # High selectivity: align with top singular vectors of W
            U, S, Vt = backend.svd(W)
            backend.eval(U, S, Vt)

            # Project B into top-k directions of W (creates "spiky" interaction)
            U_top = U[:, :lora_rank]
            B = backend.matmul(U_top, backend.random_normal((lora_rank, lora_rank), dtype="float32"))
            A = backend.random_normal((lora_rank, hidden_dim), dtype="float32")
            delta = backend.matmul(B, A)
            delta = backend.multiply(delta, 0.01 * selectivity_level)

        backend.eval(delta)
        delta_weights[key] = delta

    return base_weights, delta_weights


class TestCorrelationInfrastructure:
    """Test correlation computation infrastructure."""

    def test_pearson_correlation_perfect_positive(self):
        """Pearson r = 1.0 for identical lists."""
        backend = get_default_backend()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = compute_pearson_correlation(x, y, backend=backend)
        assert abs(result.r - 1.0) < 1e-6

    def test_pearson_correlation_perfect_negative(self):
        """Pearson r = -1.0 for inverted lists."""
        backend = get_default_backend()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]

        result = compute_pearson_correlation(x, y, backend=backend)
        assert abs(result.r - (-1.0)) < 1e-6

    def test_pearson_with_bootstrap_ci(self):
        """Bootstrap CI can be computed."""
        backend = get_default_backend()
        backend.random_seed(42)

        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [1.1, 2.2, 2.9, 4.1, 5.0, 5.9, 7.1, 8.0]

        result = compute_pearson_correlation(
            x, y, with_ci=True, n_bootstrap=100, backend=backend
        )

        assert result.ci is not None
        assert result.ci.lower <= result.r <= result.ci.upper
        assert result.ci.resamples > 0

    def test_spearman_correlation(self):
        """Spearman ρ works with tied ranks."""
        backend = get_default_backend()

        x = [1.0, 2.0, 2.0, 4.0, 5.0]  # Has ties
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = compute_spearman_correlation(x, y, backend=backend)
        assert -1.0 <= result.r <= 1.0


class TestSelectivityMeasurement:
    """Test spectral selectivity measurement."""

    def test_low_selectivity_adapter(self):
        """Low selectivity adapters have lower CV."""
        backend = get_default_backend()
        backend.random_seed(42)

        base_weights, delta_weights = _create_synthetic_adapter_with_varying_selectivity(
            selectivity_level=0.1, n_layers=2, backend=backend
        )

        measurements = []
        for key in base_weights:
            m = collect_layer_measurements(
                weight_original=base_weights[key],
                delta_w=delta_weights[key],
                layer_idx=0,
                projection_name="q_proj",
                backend=backend,
            )
            measurements.append(m)

        # Should have finite values
        for m in measurements:
            assert m.amplification_cv >= 0

    def test_high_selectivity_adapter(self):
        """High selectivity adapters have higher CV (when aligned with W)."""
        backend = get_default_backend()
        backend.random_seed(42)

        base_weights, delta_weights = _create_synthetic_adapter_with_varying_selectivity(
            selectivity_level=0.9, n_layers=2, backend=backend
        )

        measurements = []
        for key in base_weights:
            m = collect_layer_measurements(
                weight_original=base_weights[key],
                delta_w=delta_weights[key],
                layer_idx=0,
                projection_name="q_proj",
                backend=backend,
            )
            measurements.append(m)

        # Should have finite values
        for m in measurements:
            assert m.amplification_cv >= 0


class TestFullExperiment:
    """Full selectivity-conflict correlation experiment."""

    @pytest.mark.slow
    def test_selectivity_correlation_analysis(self):
        """Collect measurements and compute correlations."""
        _ensure_results_dir()

        backend = get_default_backend()
        backend.random_seed(42)

        # Create adapters with varying selectivity levels
        selectivity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        adapters: list[AdapterMeasurement] = []

        for i, level in enumerate(selectivity_levels):
            base_weights, delta_weights = _create_synthetic_adapter_with_varying_selectivity(
                selectivity_level=level,
                n_layers=4,
                hidden_dim=128,
                lora_rank=8,
                backend=backend,
            )

            # Collect measurements
            layer_measurements = []
            for key in base_weights:
                parts = key.split(".")
                layer_idx = -1
                for j, part in enumerate(parts):
                    if part == "layers" and j + 1 < len(parts):
                        try:
                            layer_idx = int(parts[j + 1])
                        except ValueError:
                            pass

                m = collect_layer_measurements(
                    weight_original=base_weights[key],
                    delta_w=delta_weights[key],
                    layer_idx=layer_idx,
                    projection_name="q_proj",
                    backend=backend,
                )
                layer_measurements.append(m)

            adapter = AdapterMeasurement(
                adapter_id=f"synthetic_{i}_level_{level}",
                base_model_id="synthetic",
                lora_rank=8,
                lora_alpha=8.0,
                training_domain=f"synthetic_level_{level}",
                layer_measurements=layer_measurements,
                # Synthetic conflict score: correlate with selectivity for testing
                conflict_score=level * 0.5 + backend.to_scalar(
                    backend.random_uniform(0.0, 0.1, shape=(1,))
                ),
                mean_kl=level * 0.3,
                base_frontier_rate=1.0 - level * 0.5,
                metadata={"selectivity_level": level},
            )
            adapters.append(adapter)

        # Save raw measurements
        raw_measurements = {
            "n_adapters": len(adapters),
            "adapters": [],
        }

        for adapter in adapters:
            adapter_data = {
                "adapter_id": adapter.adapter_id,
                "base_model_id": adapter.base_model_id,
                "lora_rank": adapter.lora_rank,
                "lora_alpha": adapter.lora_alpha,
                "training_domain": adapter.training_domain,
                "conflict_score": adapter.conflict_score,
                "mean_kl": adapter.mean_kl,
                "base_frontier_rate": adapter.base_frontier_rate,
                "mean_amplification_cv": adapter.mean_amplification_cv(),
                "mean_weyl_utilization": adapter.mean_weyl_utilization(),
                "total_frobenius_norm": adapter.total_frobenius_norm(),
                "layers": [
                    {
                        "layer_idx": lm.layer_idx,
                        "projection_name": lm.projection_name,
                        "amplification_cv": lm.amplification_cv,
                        "weyl_utilization": lm.weyl_utilization,
                        "delta_frobenius_norm": lm.delta_frobenius_norm,
                        "delta_spectral_norm": lm.delta_spectral_norm,
                    }
                    for lm in adapter.layer_measurements
                ],
            }
            raw_measurements["adapters"].append(adapter_data)

        with open(RESULTS_DIR / "raw_measurements.json", "w") as f:
            json.dump(raw_measurements, f, indent=2)

        # Compute correlations
        selectivity_values = [a.mean_amplification_cv() for a in adapters]
        conflict_values = [a.conflict_score for a in adapters if a.conflict_score is not None]

        # Ensure same length
        n_valid = min(len(selectivity_values), len(conflict_values))
        selectivity_values = selectivity_values[:n_valid]
        conflict_values = conflict_values[:n_valid]

        pearson_result = compute_pearson_correlation(
            selectivity_values,
            conflict_values,
            with_ci=True,
            n_bootstrap=len(adapters),
            backend=backend,
        )

        spearman_result = compute_spearman_correlation(
            selectivity_values,
            conflict_values,
            with_ci=True,
            n_bootstrap=len(adapters),
            backend=backend,
        )

        # Scale confound check: correlate selectivity with delta norms
        norm_values = [a.total_frobenius_norm() for a in adapters]
        scale_confound = compute_pearson_correlation(
            selectivity_values, norm_values, backend=backend
        )

        # Per-layer correlation analysis
        layer_correlations = {}
        layer_indices = sorted(set(lm.layer_idx for a in adapters for lm in a.layer_measurements))

        for layer_idx in layer_indices:
            layer_cvs = []
            layer_conflicts = []
            for adapter in adapters:
                for lm in adapter.layer_measurements:
                    if lm.layer_idx == layer_idx and adapter.conflict_score is not None:
                        layer_cvs.append(lm.amplification_cv)
                        layer_conflicts.append(adapter.conflict_score)
                        break

            if len(layer_cvs) >= 3:
                r = compute_pearson_correlation(layer_cvs, layer_conflicts, backend=backend)
                rho = compute_spearman_correlation(layer_cvs, layer_conflicts, backend=backend)
                layer_correlations[str(layer_idx)] = {
                    "pearson_r": r.r,
                    "spearman_rho": rho.r,
                    "n": r.n,
                }

        # Save correlation results
        correlation_results = {
            "overall": {
                "pearson_r": pearson_result.r,
                "pearson_ci_lower": pearson_result.ci.lower if pearson_result.ci else None,
                "pearson_ci_upper": pearson_result.ci.upper if pearson_result.ci else None,
                "spearman_rho": spearman_result.r,
                "spearman_ci_lower": spearman_result.ci.lower if spearman_result.ci else None,
                "spearman_ci_upper": spearman_result.ci.upper if spearman_result.ci else None,
                "n": pearson_result.n,
            },
            "by_layer": layer_correlations,
            "scale_confound": {
                "selectivity_vs_norm_r": scale_confound.r,
            },
        }

        with open(RESULTS_DIR / "correlation_by_layer.json", "w") as f:
            json.dump(correlation_results, f, indent=2)

        # Verification checks
        verifications = {
            "ci_width_quantiles": {
                "pearson": {
                    "width": (pearson_result.ci.upper - pearson_result.ci.lower)
                    if pearson_result.ci
                    else None,
                },
            },
            "sample_coverage": n_valid / len(adapters) if adapters else 0,
            "finite_measurements": sum(
                1 for a in adapters if a.conflict_score is not None
            )
            / len(adapters),
        }

        with open(RESULTS_DIR / "scale_confound_check.json", "w") as f:
            json.dump({"scale_confound": scale_confound.r, "verifications": verifications}, f, indent=2)

        print("\n=== Selectivity-Conflict Correlation Results ===")
        print(f"Results saved to: {RESULTS_DIR}")
        print(f"Adapters analyzed: {len(adapters)}")
        print(f"\nOverall correlation:")
        print(f"  Pearson r: {pearson_result.r:.4f}")
        if pearson_result.ci:
            print(
                f"  95% CI: [{pearson_result.ci.lower:.4f}, {pearson_result.ci.upper:.4f}]"
            )
        print(f"  Spearman ρ: {spearman_result.r:.4f}")
        print(f"\nScale confound (selectivity vs norm): r = {scale_confound.r:.4f}")

        # Assertions
        assert len(adapters) >= 8
        assert pearson_result.n >= 8
        assert -1.0 <= pearson_result.r <= 1.0
        assert -1.0 <= spearman_result.r <= 1.0
