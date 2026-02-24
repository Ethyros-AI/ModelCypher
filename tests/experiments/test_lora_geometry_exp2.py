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

"""Experiment 2: Four-Condition Measurement

Question: How do geometric metrics distribute across training conditions?

Conditions:
1. UNTRAINED: LoRA initialized, 0 training steps
2. TRAINED: Normal LoRA training, converged (requires actual trained adapters)
3. RANDOM_LABELS: Trained on shuffled labels (requires actual training)
4. PURE_RANDOM: Random B, A matrices (no training)

This experiment establishes baseline metric distributions.

Run with:
    poetry run pytest tests/experiments/test_lora_geometry_exp2.py -v -s --capture=no
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.lora_geometry.four_condition import (
    ConditionType,
    FourConditionExperiment,
    create_four_condition_synthetic,
)
from modelcypher.experimental.lora_geometry.statistics import (
    compute_bootstrap_ci,
    compute_permutation_test,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# Results directory
RESULTS_DIR = Path("results/four_condition")


def _ensure_results_dir() -> None:
    """Create results directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _create_synthetic_base_weights(
    n_layers: int = 4,
    hidden_dim: int = 256,
    backend: "Backend | None" = None,
) -> dict[str, "Array"]:
    """Create synthetic base weights for testing."""
    if backend is None:
        backend = get_default_backend()

    weights = {}
    projections = ["q_proj", "k_proj", "v_proj", "o_proj"]

    for layer_idx in range(n_layers):
        for proj in projections:
            key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            w = backend.random_normal((hidden_dim, hidden_dim), dtype="float32")
            backend.eval(w)
            weights[key] = w

    return weights


class TestFourConditionSynthetic:
    """Test synthetic condition creation and measurement."""

    def test_create_untrained_adapters(self):
        """UNTRAINED adapters have near-zero delta norms."""
        backend = get_default_backend()
        backend.random_seed(42)

        base_weights = _create_synthetic_base_weights(n_layers=2, backend=backend)

        experiment = create_four_condition_synthetic(
            base_weights=base_weights,
            base_model_id="synthetic_test",
            adapters_per_condition=4,
            lora_rank=8,
            backend=backend,
        )

        # Check UNTRAINED
        untrained = experiment.measurements.get(ConditionType.UNTRAINED, [])
        assert len(untrained) == 4

        # Untrained should have near-zero delta norms (B @ zeros = 0)
        for m in untrained:
            total_norm = m.total_frobenius_norm()
            # Should be exactly 0 or very close (numerical noise)
            assert total_norm < 1e-6, f"Untrained adapter has norm {total_norm}"

    def test_create_pure_random_adapters(self):
        """PURE_RANDOM adapters have non-zero delta norms."""
        backend = get_default_backend()
        backend.random_seed(42)

        base_weights = _create_synthetic_base_weights(n_layers=2, backend=backend)

        experiment = create_four_condition_synthetic(
            base_weights=base_weights,
            base_model_id="synthetic_test",
            adapters_per_condition=4,
            lora_rank=8,
            scale=0.1,
            backend=backend,
        )

        # Check PURE_RANDOM
        pure_random = experiment.measurements.get(ConditionType.PURE_RANDOM, [])
        assert len(pure_random) == 4

        # Random should have non-zero norms
        for m in pure_random:
            total_norm = m.total_frobenius_norm()
            assert total_norm > 0, "Random adapter has zero norm"

    def test_metric_extraction(self):
        """Metrics can be extracted by condition."""
        backend = get_default_backend()
        backend.random_seed(42)

        base_weights = _create_synthetic_base_weights(n_layers=2, backend=backend)

        experiment = create_four_condition_synthetic(
            base_weights=base_weights,
            base_model_id="synthetic_test",
            adapters_per_condition=4,
            lora_rank=8,
            backend=backend,
        )

        # Extract metrics
        cv_by_condition = experiment.get_metric_by_condition("amplification_cv")
        experiment.get_metric_by_condition("weyl_utilization")
        experiment.get_metric_by_condition("delta_frobenius_norm")

        # UNTRAINED should be present
        assert ConditionType.UNTRAINED in cv_by_condition
        assert ConditionType.PURE_RANDOM in cv_by_condition

        # Should have 4 values per condition
        assert len(cv_by_condition[ConditionType.PURE_RANDOM]) == 4

    def test_per_layer_metrics(self):
        """Per-layer metrics can be extracted."""
        backend = get_default_backend()
        backend.random_seed(42)

        base_weights = _create_synthetic_base_weights(n_layers=2, backend=backend)

        experiment = create_four_condition_synthetic(
            base_weights=base_weights,
            base_model_id="synthetic_test",
            adapters_per_condition=2,
            lora_rank=8,
            backend=backend,
        )

        # Get per-layer metrics for PURE_RANDOM
        layer_cv = experiment.get_per_layer_metrics(
            ConditionType.PURE_RANDOM, "amplification_cv"
        )

        # Should have layer indices 0 and 1
        assert 0 in layer_cv or 1 in layer_cv


class TestPermutationAnalysis:
    """Test permutation test infrastructure."""

    def test_permutation_test_same_distribution(self):
        """Permutation test with same distribution gives high p-value."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Same distribution
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [1.5, 2.5, 3.5, 4.5, 5.5]

        result = compute_permutation_test(
            group1, group2, n_permutations=100, backend=backend
        )

        # p-value should be high (no difference)
        assert result.p_value > 0.05
        assert result.n_permutations > 0

    def test_permutation_test_different_distribution(self):
        """Permutation test with different distributions gives low p-value."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Very different distributions
        group1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        group2 = [100.0, 101.0, 102.0, 103.0, 104.0]

        result = compute_permutation_test(
            group1, group2, n_permutations=100, backend=backend
        )

        # p-value should be low (clear difference)
        assert result.p_value < 0.1
        assert result.observed_stat != 0


class TestFullExperiment:
    """Full experiment with measurement and analysis."""

    @pytest.mark.slow
    def test_full_four_condition_experiment(self):
        """Run full four-condition experiment with synthetic data."""
        _ensure_results_dir()

        backend = get_default_backend()
        backend.random_seed(42)

        # Create base weights (small for testing)
        base_weights = _create_synthetic_base_weights(
            n_layers=4, hidden_dim=128, backend=backend
        )

        # Create experiment (UNTRAINED and PURE_RANDOM only for synthetic)
        experiment = create_four_condition_synthetic(
            base_weights=base_weights,
            base_model_id="synthetic_test_full",
            adapters_per_condition=8,  # Per plan: 8 adapters per condition
            lora_rank=8,
            scale=0.01,
            backend=backend,
        )

        # Collect raw measurements
        raw_measurements = {
            "base_model_id": experiment.base_model_id,
            "adapters_per_condition": experiment.adapters_per_condition,
            "conditions": {},
        }

        for condition, measurements in experiment.measurements.items():
            condition_data = []
            for m in measurements:
                adapter_data = {
                    "adapter_id": m.adapter_id,
                    "lora_rank": m.lora_rank,
                    "lora_alpha": m.lora_alpha,
                    "training_domain": m.training_domain,
                    "mean_amplification_cv": m.mean_amplification_cv(),
                    "mean_weyl_utilization": m.mean_weyl_utilization(),
                    "total_frobenius_norm": m.total_frobenius_norm(),
                    "layers": [
                        {
                            "layer_idx": lm.layer_idx,
                            "projection_name": lm.projection_name,
                            "amplification_cv": lm.amplification_cv,
                            "weyl_utilization": lm.weyl_utilization,
                            "delta_frobenius_norm": lm.delta_frobenius_norm,
                            "delta_spectral_norm": lm.delta_spectral_norm,
                        }
                        for lm in m.layer_measurements
                    ],
                }
                condition_data.append(adapter_data)
            raw_measurements["conditions"][condition.value] = condition_data

        # Save raw measurements
        with open(RESULTS_DIR / "raw_measurements.json", "w") as f:
            json.dump(raw_measurements, f, indent=2)

        # Permutation tests between conditions
        permutation_results = {}

        conditions = list(experiment.measurements.keys())
        if len(conditions) >= 2:
            for metric_name in ["amplification_cv", "weyl_utilization"]:
                metrics = experiment.get_metric_by_condition(metric_name)

                # Test UNTRAINED vs PURE_RANDOM
                if (
                    ConditionType.UNTRAINED in metrics
                    and ConditionType.PURE_RANDOM in metrics
                ):
                    untrained_vals = metrics[ConditionType.UNTRAINED]
                    random_vals = metrics[ConditionType.PURE_RANDOM]

                    if untrained_vals and random_vals:
                        result = compute_permutation_test(
                            untrained_vals,
                            random_vals,
                            n_permutations=1000,
                            backend=backend,
                        )

                        permutation_results[
                            f"{metric_name}_untrained_vs_pure_random"
                        ] = {
                            "observed_diff": result.observed_stat,
                            "p_value": result.p_value,
                            "percentile_5": result.percentile_5,
                            "percentile_50": result.percentile_50,
                            "percentile_95": result.percentile_95,
                            "n_permutations": result.n_permutations,
                        }

        # Save permutation test results
        with open(RESULTS_DIR / "permutation_tests.json", "w") as f:
            json.dump(permutation_results, f, indent=2)

        # Delta norm by condition
        norm_by_condition = {}
        for condition, measurements in experiment.measurements.items():
            norms = [m.total_frobenius_norm() for m in measurements]
            if norms:
                norm_by_condition[condition.value] = {
                    "min": min(norms),
                    "median": sorted(norms)[len(norms) // 2],
                    "max": max(norms),
                    "values": norms,
                }

        with open(RESULTS_DIR / "delta_norm_by_condition.json", "w") as f:
            json.dump(norm_by_condition, f, indent=2)

        # Verification checks
        verifications = {
            "ci_width_quantiles": {},
            "sample_coverage": 1.0,  # All synthetic, no NaN
            "delta_norm_distribution": norm_by_condition,
            "permutation_test_validity": {},
        }

        # Check permutation test validity
        for key, perm_result in permutation_results.items():
            verifications["permutation_test_validity"][key] = {
                "n_unique_permutations": perm_result["n_permutations"],
                "sufficient": perm_result["n_permutations"] >= 50,
            }

        with open(RESULTS_DIR / "verification_checks.json", "w") as f:
            json.dump(verifications, f, indent=2)

        print("\n=== Four-Condition Experiment Results ===")
        print(f"Results saved to: {RESULTS_DIR}")
        print(f"Conditions tested: {[c.value for c in experiment.measurements.keys()]}")

        # Report key findings
        for condition, measurements in experiment.measurements.items():
            cvs = [m.mean_amplification_cv() for m in measurements]
            weyls = [m.mean_weyl_utilization() for m in measurements]
            print(f"\n{condition.value}:")
            print(f"  Amplification CV: mean={sum(cvs)/len(cvs):.4f}")
            print(f"  Weyl Utilization: mean={sum(weyls)/len(weyls):.4f}")

        # Assertions
        assert len(experiment.measurements) >= 2
        assert ConditionType.UNTRAINED in experiment.measurements
        assert ConditionType.PURE_RANDOM in experiment.measurements
